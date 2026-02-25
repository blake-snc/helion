from __future__ import annotations

import ast
import contextlib
from itertools import dropwhile
import logging
from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence
from typing import cast

log = logging.getLogger(__name__)

import sympy
import torch
from torch._inductor import dependencies
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import OutputSpec
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import TemplateKernelMetadata
from torch._inductor.ir import ResolvedEpilogueSpec
from torch._inductor.ir import ResolvedPrologueSpec
from torch._inductor.ir import TensorBox
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.lowering import register_lowering
from torch._inductor.ir import pointwise_uses_index_expr
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
from ..ast_extension import unparse
from ..ast_read_writes import ast_rename
from ..generate_ast import generate_ast
from ..indexing_strategy import SubscriptIndexing
from ..output_header import get_needed_imports
from ..output_header import library_imports

if TYPE_CHECKING:
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.ir import MultiOutput
    from torch._inductor.scheduler import BaseSchedulerNode

    from ..inductor_lowering import CodegenState
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


class _CodeExpr(str):
    """A str whose repr() returns itself, for embedding variable names in generated code.

    When generating a kernel call like ``kernel(x, (a, b))``, container args are
    rebuilt via pytree into e.g. ``(_CodeExpr("a"), _CodeExpr("b"))``.  Python's
    built-in ``repr()`` on that tuple then produces ``(a, b)`` instead of
    ``('a', 'b')``, giving us correct code for free.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Inductor template buffer for Helion kernel."""

    def __init__(
        self,
        layout: OutputSpec,
        inputs: Sequence[IRNode],
        kernel: Kernel,
        constant_args: dict[str, object],
        tensor_arg_names: list[str],
        bound_kernel: BoundKernel,
        mutated_input_names: list[str] | None = None,
        autotune_args: tuple[object, ...] | None = None,
    ) -> None:
        # Required by Inductor scheduler
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        self.named_input_nodes = dict(zip(tensor_arg_names, inputs, strict=True))
        self.kernel_name: str | None = None
        self._helion_kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._autotune_args = autotune_args

        # Maps output buffer name -> (kernel param name, MultiOutput indices).
        self._output_buf_to_param: dict[
            str, tuple[str | None, list[tuple[type, int]]]
        ] = {}
        # Pre-computed metadata set by lower_helion_kernel after MultiOutput creation.
        self._template_metadata: TemplateKernelMetadata | None = None

        mutated_inputs_irnodes = [
            self.named_input_nodes[n]
            for n in (mutated_input_names or [])
            if n in self.named_input_nodes
        ] or None

        super().__init__(
            layout=cast("Layout", layout),
            inputs=inputs,
            make_kernel_render=lambda tb, hint_override=None: (self, self.render),
            mutated_inputs=mutated_inputs_irnodes,
            # Mark all inputs as eligible for prologue fusion;
            # the scheduler decides which ones actually get fused.
            allowed_prologue_inps=OrderedSet(
                inp.get_name()
                for inp in inputs  # type: ignore[union-attr]
            ),
        )

        for inp in mutated_inputs_irnodes or []:
            if hasattr(inp, "get_name"):
                V.graph.never_reuse_buffers.add(inp.get_name())


    # Layout is always MultiOutputLayout: reads from inputs only,
    # writes go through MultiOutput children, no allocation needed.

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype for prologue-fusion heuristic checks.

        The parent TemplateBuffer.dtype does ``self.get_layout().dtype``,
        but our layout is MultiOutputLayout which has no dtype attribute.
        We override to infer dtype from the first input tensor instead.
        """
        if self.inputs:
            return self.inputs[0].get_dtype()  # type: ignore[union-attr]
        return torch.float32

    @property
    def template_metadata(self) -> TemplateKernelMetadata:
        """Pre-computed metadata implementing Inductor's TemplateKernelMetadata contract.

        Overrides TritonTemplateBuffer.template_metadata to return the metadata
        computed in lower_helion_kernel (after MultiOutput nodes are created),
        avoiding re-querying Inductor IR at codegen time.
        """
        assert self._template_metadata is not None, "metadata set by lower_helion_kernel"
        return self._template_metadata

    def get_template_output_buf_names(self) -> set[str]:
        """Return names of all MultiOutput buffers produced by this kernel."""
        return self._template_metadata.all_output_names if self._template_metadata is not None else set(self._output_buf_to_param)

    def get_output_param_mapping(self) -> dict[str, str]:
        """Return {output_buf_name: kernel_param_name} for all fusable outputs."""
        return self._template_metadata.fusable_outputs if self._template_metadata is not None else {
            buf: val[0]
            for buf, val in self._output_buf_to_param.items()
            if val[0] is not None
        }

    def get_input_param_mapping(self) -> dict[str, str]:
        """Return {input_buf_name: kernel_param_name} for all fusable inputs."""
        return self._template_metadata.all_inputs if self._template_metadata is not None else {
            inp.get_name(): param
            for param, inp in self.named_input_nodes.items()
        }

    def is_fusable_epilogue_output(self, output_buf_name: str) -> bool:
        """Template-specific epilogue eligibility check."""
        return output_buf_name in (self._template_metadata.fusable_outputs if self._template_metadata is not None else self.get_output_param_mapping())

    def should_allocate(self) -> bool:
        return False

    def get_size(self) -> Sequence[sympy.Expr]:
        return []

    def _generate_triton_ast(self) -> ast.Module | None:
        """Generate and rename the Triton kernel AST.

        Returns the AST with function names replaced by Placeholder.KERNEL_NAME,
        or None if the bound kernel is not available.
        """
        if not self._bound_kernel:
            return None
        # Ensure config is available (triggers autotuning if needed)
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)
        cfg = self._bound_kernel._config
        assert cfg is not None, "Config should be set after ensure_config_exists"
        host_fn = self._helion_kernel.name
        inner_fn = f"_helion_{host_fn}"
        inner_fn_placeholder = f"{inner_fn}_{Placeholder.KERNEL_NAME}"

        # Generate Python AST for Triton kernel
        with self._bound_kernel.env:
            host_function = self._bound_kernel.host_function
            assert host_function is not None, "BoundKernel must have a host_function"
            root = generate_ast(
                host_function,
                cfg,
                emit_repro_caller=False,
                store_transform=self._codegen_epilogue_fusion if self._epilogue_specs else None,
                load_transform=self._codegen_prologue_fusion if self._prologue_specs else None,
            )

        # Collect module-level variable names that need uniquification
        # (constexpr assignments like _BLOCK_SIZE_0 = tl.constexpr(32))
        assert isinstance(root, ast.Module)
        module_level_vars: dict[str, str] = {}
        for node in root.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_level_vars[target.id] = (
                            f"{target.id}_{Placeholder.KERNEL_NAME}"
                        )

        # Rename functions, module-level vars, and update references
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                if node.name == host_fn:
                    node.name = str(Placeholder.KERNEL_NAME)
                elif node.name == inner_fn:
                    node.name = inner_fn_placeholder
            elif isinstance(node, ast.Name):
                if node.id == inner_fn:
                    node.id = inner_fn_placeholder
                elif node.id in module_level_vars:
                    node.id = module_level_vars[node.id]

        return root  # pyrefly: ignore[bad-return]

    def _ast_to_source(self, root: ast.Module) -> str:
        """Convert AST to source code with imports."""
        return get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )

    def render(self) -> PartialRender:
        """Generate Triton code."""
        root = self._generate_triton_ast()
        if root is None:
            return PartialRender("", {})
        return PartialRender(self._ast_to_source(root), {})

    def call_kernel(
        self, kernel_name: str, template_buffer: TritonTemplateBuffer | None = None
    ) -> None:
        """Emit the kernel call site."""
        wrapper = V.graph.wrapper_code
        output_name = self.get_name()
        reinterp_count = 0

        def get_input_expr(arg_name: str, inp: IRNode) -> str:
            nonlocal reinterp_count
            buf_name = inp.get_name()  # type: ignore[union-attr]
            pro_spec = self._prologue_specs.get(arg_name)
            source_buf = pro_spec.source_buf if pro_spec is not None else None

            if source_buf is not None:
                # This input's buffer is prologue-fused: use source buffer.
                if isinstance(inp, ReinterpretView):
                    # Preserve the view (strides/offsets) but point to source
                    sizes = tuple(inp.get_size())
                    strides = tuple(inp.get_stride())
                    offset = inp.layout.offset
                    expr = f"reinterpret_tensor({source_buf}, {sizes}, {strides}, {offset})"
                    wrapper.writeline(f"reinterp_{reinterp_count} = {expr}")
                    result = f"reinterp_{reinterp_count}"
                    reinterp_count += 1
                    return result
                return source_buf

            if not isinstance(inp, ReinterpretView):
                return buf_name
            expr = wrapper.codegen_reinterpret_view(
                inp.data,
                list(inp.get_size()),
                list(inp.get_stride()),
                inp.layout.offset,
                wrapper.writeline,
            )
            if expr != inp.data.get_name():
                wrapper.writeline(f"reinterp_{reinterp_count} = {expr}")
                expr = f"reinterp_{reinterp_count}"
                reinterp_count += 1
            return expr

        arg_inputs = {
            name: get_input_expr(name, inp)
            for name, inp in self.named_input_nodes.items()
        }

        all_args: dict[str, object] = {n: _CodeExpr(v) for n, v in arg_inputs.items()}
        for n, v in self._constant_args_dict.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        sig = self._helion_kernel.signature.parameters
        args = [
            repr(all_args[n]) if n in all_args else repr(p.default)
            for n, p in sig.items()
            if n in all_args or p.default is not p.empty
        ]

        # Add epilogue extra parameters (outputs and inputs)
        args.extend(buf_name for _, buf_name in self._epilogue_extra_params)
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Emit MultiOutput extraction code for each output buffer.
        # MultiOutput nodes are marked as run by codegen_with_fusion (via Inductor),
        # so their separate codegen is suppressed. We must emit the extraction
        # here so that downstream consumers can reference the buffer names.
        for mo_name, (_param, indices) in sorted(self._output_buf_to_param.items()):
            if mo_name not in self.removed_buffers:
                idx_str = output_name
                for _, idx in indices:
                    idx_str = f"{idx_str}[{idx}]"
                wrapper.writeline(f"{mo_name} = {idx_str}")

    def codegen_with_fusion(
        self,
        epilogue_specs: list[ResolvedEpilogueSpec],
        prologue_specs: list[ResolvedPrologueSpec],
        extra_params: list[tuple[str, str]],
        render: Callable[[], PartialRender | str],
    ) -> str:
        """Single entry point from Inductor. Generates the complete Triton kernel source.

        Called by ``_codegen_single_template`` after Inductor has resolved fusion
        specs into Triton expression strings (via ``_resolve_epilogue_specs`` /
        ``_resolve_prologue_specs``).  Always returns a source string; Inductor
        handles ``define_kernel`` and ``mark_run``.

        Fusion path (epilogue_specs or prologue_specs non-empty):
          1. Autotune to fix config before specs are embedded.
          2. Store specs in ``_epilogue_specs`` / ``_prologue_specs`` via the
             base-class helper.  ``param_alias_map`` is passed so that the same
             input buffer appearing under multiple param names (e.g. ``k_add(x, x)``)
             is handled generically by Inductor rather than requiring a Helion
             extension.
          3. Regenerate the Triton AST.  ``_generate_triton_ast`` detects non-empty
             spec dicts and activates ``store_transform`` / ``load_transform``
             callbacks so ``hl.store`` / ``hl.load`` inline the fused expressions.
          4. Inject extra params (fusion inputs/outputs) into the inner function,
             host function, and launcher call.

        No-fusion path (empty specs): skip straight to AST generation and
        serialisation.  ``benchmark_kernel`` wrapping is intentionally omitted
        because ``HelionTemplateBuffer`` is not a ``SIMDKernel``.
        """
        if epilogue_specs or prologue_specs:
            # Autotune must fire *before* fusion specs are built so the autotuned
            # (unfused) code does not contain fusion patterns.  _generate_triton_ast
            # will skip ensure_config_exists when config is already set.
            if self._autotune_args and self._bound_kernel:
                self._bound_kernel.ensure_config_exists(self._autotune_args)

            # Build param alias map: buf_name -> [param1, param2, ...].
            # Passed to _apply_fusion_specs_to_layout so that the base class
            # registers each prologue spec under every aliased parameter name.
            # This handles the k_add(x, x) case generically in Inductor without
            # requiring a Helion-specific extension after the base-class call.
            input_buf_to_params: dict[str, list[str]] = {}
            for param_name, inp in self.named_input_nodes.items():
                input_buf_to_params.setdefault(inp.get_name(), []).append(param_name)  # type: ignore[union-attr]

            self._apply_fusion_specs_to_layout(
                epilogue_specs,
                prologue_specs,
                extra_params,
                param_alias_map=input_buf_to_params,
            )

        # Regenerate Triton AST.  When _epilogue_specs / _prologue_specs are
        # non-empty, _generate_triton_ast activates the store/load transform
        # callbacks so hl.store / hl.load emit fused expressions inline.
        root = self._generate_triton_ast()
        if root is None:
            return ""

        # Inject extra params (epilogue inputs/outputs) into the inner function,
        # host function, and launcher call.  Helion always generates the inner
        # (device) function first, then the host wrapper — verified below.
        if self._epilogue_extra_params:
            funcs = [
                n for n in ast.iter_child_nodes(root) if isinstance(n, ast.FunctionDef)
            ]
            if len(funcs) < 2:
                raise RuntimeError(
                    f"Expected at least 2 function defs (inner + host) in generated "
                    f"Triton AST, but found {len(funcs)}. This may indicate a change "
                    f"in Helion's code generation structure."
                )
            inner_func, host_func = funcs[0], funcs[1]

            # Find the launcher call in the host function.
            launcher_call = next(
                (
                    n
                    for n in ast.walk(host_func)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("_launcher", "_default_launcher")
                ),
                None,
            )
            # Append extra parameters to inner function, host function, and launcher.
            extra_param_names = [p for p, _ in self._epilogue_extra_params]
            for name in extra_param_names:
                inner_func.args.args.append(ast.arg(arg=name))
                host_func.args.args.append(ast.arg(arg=name))
                if launcher_call is not None:
                    launcher_call.args.append(ast.Name(id=name, ctx=ast.Load()))

            # Apply redirect renames so tensor_descriptor params are renamed correctly.
            if self._epilogue_renames:
                func_params = {arg.arg for arg in inner_func.args.args}
                for orig_param, new_param in self._epilogue_renames.items():
                    if orig_param not in func_params:
                        raise RuntimeError(
                            f"Epilogue redirect expected '{orig_param}' to be a "
                            f"parameter of the Triton inner function, but found "
                            f"only: {sorted(func_params)}.  This indicates a "
                            f"mismatch between the kernel's output param naming "
                            f"and the epilogue spec's kernel_output_param."
                        )
                    ast_rename(inner_func, {orig_param: new_param})

        return self._ast_to_source(root)

    def emit_kernel_override(
        self,
        wrapper: PythonWrapperCodegen,
        src_code: str,
        kernel_name: str,
        node_schedule: Sequence[BaseSchedulerNode | object],
        kernel_path: str,
        get_kernel_metadata: Callable[
            [Sequence[BaseSchedulerNode | object], PythonWrapperCodegen],
            tuple[str, str],
        ],
    ) -> bool:
        """Entry point for kernel emission."""
        required = ("triton", "tl", "_default_launcher")
        conditional = ("libdevice", "tl_math", "triton_helpers", "helion", "hl")
        for name in (*required, *(n for n in conditional if f"{n}." in src_code)):
            wrapper.add_import_once(library_imports[name])

        # Add imports for captured global variables (e.g., "import __main__ as _source_module")
        # These are tracked in HostFunction.global_imports during kernel compilation
        if self._bound_kernel.host_function is not None:
            for imp in self._bound_kernel.host_function.global_imports.values():
                wrapper.add_import_once(imp.codegen())

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        # Skip import lines at the beginning
        for line in dropwhile(
            lambda ln: (
                (s := ln.strip()).startswith(("from __future__", "import ", "from "))
                or not s
            ),
            src_code.split("\n"),
        ):
            wrapper.header.writeline(line)
        wrapper.header.writeline("")
        return True

    def _codegen_epilogue_fusion(
        self,
        state: "CodegenState",
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        """Apply resolved epilogue spec at an ``hl.store`` site during Triton codegen.

        Called from memory_ops.py when epilogue fusion is active.  Resolves the
        Helion-specific indexing context (offset, mask, per-dimension index
        expressions) and delegates the generic expression-application logic to
        ``TritonTemplateBuffer.apply_resolved_epilogue_at_store``, which is
        reusable across Triton-generating backends.
        """
        assert self._epilogue_specs
        param_name = state.device_function.tensor_arg(tensor).name
        spec = self._epilogue_specs.get(param_name)
        if spec is None:
            return value

        # Unique per-epilogue name avoids Triton type conflicts across branches.
        epi_idx = list(self._epilogue_specs.keys()).index(param_name)
        kernel_val_name = f"_kernel_val_{epi_idx}"

        # Compute Helion-specific indexing context for this store site.
        indexing = SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
        offset_str = ast.unparse(indexing.index_expr)
        mask_str = ast.unparse(indexing.mask_expr) if indexing.has_mask() else None

        # Delegate generic expression-application logic to the Inductor base class.
        stmts, fused_val = TritonTemplateBuffer.apply_resolved_epilogue_at_store(
            spec,
            kernel_val_name,
            value,  # type: ignore[arg-type]
            indexing.dim_index_exprs,
            offset_str,
            mask_str,
        )
        for stmt in stmts:
            state.add_statement(stmt)
        return fused_val  # type: ignore[return-value]

    def _codegen_prologue_fusion(
        self,
        state: "CodegenState",
        tensor: torch.Tensor,
        value: ast.AST,
    ) -> ast.AST:
        """Apply resolved prologue spec at an ``hl.load`` site during Triton codegen.

        Called from memory_ops.py when prologue fusion is active.  Delegates to
        ``TritonTemplateBuffer.apply_resolved_prologue_at_load``, which substitutes
        the ``_load_val`` placeholder in the pre-traced fused expression with the
        actual load AST — inlining the prologue op (e.g. dtype cast) at the load
        site.  Reusable across Triton-generating backends.
        """
        assert self._prologue_specs
        param_name = state.device_function.tensor_arg(tensor).name
        spec = self._prologue_specs.get(param_name)
        if spec is None:
            return value
        return TritonTemplateBuffer.apply_resolved_prologue_at_load(spec, value)  # type: ignore[return-value]

    def set_current_node(self, node: BaseSchedulerNode) -> contextlib.nullcontext[None]:
        """Set current node for codegen context."""
        return contextlib.nullcontext()


def _flatten_return_ast(
    ast_node: ast.expr | None,
    structured: object,
) -> list[ast.expr | None]:
    """Get the per-leaf AST nodes in DFS order matching build_multi_outputs traversal.

    Walks `structured` in the same order as build_multi_outputs to produce a
    flat list mapping leaf_idx → the corresponding AST node from return_ast.
    Used to extract kernel parameter names (ast.Name nodes) and detect
    symbolic (non-constant) non-tensor returns.
    """
    result: list[ast.expr | None] = []

    def walk(node: ast.expr | None, out: object) -> None:
        if isinstance(out, (list, tuple)):
            elts = node.elts if isinstance(node, (ast.Tuple, ast.List)) else None
            for i, item in enumerate(out):
                walk(elts[i] if elts is not None and i < len(elts) else None, item)
        else:
            result.append(node)  # leaf (tensor or non-tensor)

    walk(ast_node, structured)
    return result


@register_lowering(helion_kernel_wrapper_mutation, type_promotion_kind=None)
def lower_helion_kernel(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
) -> tuple[TensorBox, ...]:
    """Lower a Helion kernel call to HelionTemplateBuffer."""
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Realize inputs: convert TensorBox to buffer/ReinterpretView.
    # Use TritonTemplateBuffer.realize_template_input to preserve MultiOutput
    # layouts (ExternKernel.realize_input would lose non-contiguous strides).
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = TritonTemplateBuffer.realize_template_input(tb)

    # Build fake tensors for kernel binding (sympy exprs -> concrete ints)
    def as_int(x: object, default: int) -> int:
        return int(x) if isinstance(x, (int, sympy.Integer)) else default

    all_args: dict[str, object] = {**constant_args}
    for n, r in realized.items():
        all_args[n] = torch.empty_strided(
            [as_int(s, 64) for s in r.get_size()],
            [as_int(s, 1) for s in r.get_stride()],
            dtype=r.get_dtype(),
            device=r.get_device(),
        )
    _rebuild_container_args(all_args)

    fake_tensors: list[object] = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    bound = kernel.bind(tuple(fake_tensors))
    inputs = list(realized.values())

    # Derive output structure from bound kernel using inductor-time input layouts.
    # This gives correct strides even when inductor changes input memory layouts.
    flat_leaves, tree_spec, return_ast = _get_flat_output(bound.host_function)
    example_outputs = [leaf for leaf in flat_leaves if isinstance(leaf, torch.Tensor)]

    # Create buffer for scheduling
    dev = (
        example_outputs[0].device
        if example_outputs
        else inputs[0].get_device()
        if inputs
        else torch.device("cuda")
    )
    assert dev is not None
    buf = HelionTemplateBuffer(
        layout=MultiOutputLayout(device=dev),
        inputs=inputs,
        kernel=kernel,
        constant_args=constant_args,
        tensor_arg_names=list(realized.keys()),
        bound_kernel=bound,
        mutated_input_names=mutated_inputs_list or None,
        autotune_args=tuple(fake_tensors),
    )

    if not example_outputs:
        buf._template_metadata = TemplateKernelMetadata(
            all_inputs={inp.get_name(): p for p, inp in buf.named_input_nodes.items()},  # type: ignore[union-attr]
            fusable_outputs={},
            all_output_names=set(),
            mutated_input_names=mutated_inputs_list or [],
        )
        return ()

    # Reconstruct structured output and create MultiOutput nodes
    # (same pattern as FallbackKernel.generate_output in torch/_inductor/ir.py)
    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    # Flatten return_ast to index by leaf_idx (same traversal order as build_multi_outputs)
    flat_ast = _flatten_return_ast(return_ast, structured)

    output_sizes: dict[str, tuple[object, ...]] = {}
    has_symbolic_returns_flag = [False]

    def on_tensor_leaf(
        mo_name: str,
        mo: "MultiOutput",
        indices: list[tuple[type, int]],
        leaf_idx: int,
    ) -> None:
        ast_node = flat_ast[leaf_idx] if leaf_idx < len(flat_ast) else None
        buf._output_buf_to_param[mo_name] = (
            ast_node.id if isinstance(ast_node, ast.Name) else None,
            indices,
        )
        output_sizes[mo_name] = tuple(mo.get_size())

    def on_non_tensor_leaf(leaf_idx: int) -> None:
        ast_node = flat_ast[leaf_idx] if leaf_idx < len(flat_ast) else None
        if ast_node is not None and not isinstance(ast_node, ast.Constant):
            has_symbolic_returns_flag[0] = True

    result = TritonTemplateBuffer.build_multi_outputs(
        buf,
        structured,
        direct_alias_at_leaf={
            i: realized[name]
            for i, name in cast(
                "dict[int, str]", output_spec.get("direct_aliases", {})
            ).items()
            if name in realized
        },
        on_tensor_leaf=on_tensor_leaf,
        on_non_tensor_leaf=on_non_tensor_leaf,
    )

    # Compute fusable_outputs: param not None + not symbolic + shape matches an input
    input_shapes = {tuple(inp.get_size()) for inp in inputs}  # type: ignore[union-attr]
    fusable_outputs = {
        mo_name: param
        for mo_name, (param, _) in buf._output_buf_to_param.items()
        if param is not None
        and not has_symbolic_returns_flag[0]
        and (not input_shapes or output_sizes.get(mo_name, ()) in input_shapes)
    }
    buf._template_metadata = TemplateKernelMetadata(
        all_inputs={inp.get_name(): p for p, inp in buf.named_input_nodes.items()},  # type: ignore[union-attr]
        fusable_outputs=fusable_outputs,
        all_output_names=set(buf._output_buf_to_param),
        mutated_input_names=mutated_inputs_list or [],
    )

    return result


@register_lowering(helion_kernel_wrapper_functional, type_promotion_kind=None)
def lower_helion_kernel_functional(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[TensorBox, ...], dict[str, TensorBox]]:
    from torch._inductor.lowering import clone

    cloned = {
        n: clone(tb) if n in tensors_to_clone and isinstance(tb, TensorBox) else tb
        for n, tb in tensor_args.items()
    }
    outputs = lower_helion_kernel(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned,
        output_spec=output_spec,
    )
    return (outputs, {n: cloned[n] for n in tensors_to_clone if n in cloned})


