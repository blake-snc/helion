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
from torch._inductor.ir import ExternalTemplateBuffer
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


class HelionKernelBackend:
    """Helion's implementation of the ``ExternalKernelBackend`` protocol.

    Pure Helion logic — no Inductor IR node inheritance, no
    ``TritonTemplateBuffer`` subclassing.  Receives pre-resolved fusion specs
    from ``ExternalTemplateBuffer.codegen_with_fusion`` and emits Triton
    source.  The only Inductor types this class touches are the documented
    protocol dataclasses (``ResolvedEpilogueSpec``, ``ResolvedPrologueSpec``,
    ``TemplateKernelMetadata``) and the ``PythonWrapperCodegen`` API — both
    stable, intentional interfaces.

    Lifecycle
    ---------
    1. ``lower_helion_kernel`` creates a ``HelionKernelBackend`` and wraps it
       in an ``ExternalTemplateBuffer``.
    2. Inductor's scheduler queries ``get_metadata()`` to plan fusion.
    3. ``ExternalTemplateBuffer.codegen_with_fusion`` applies spec expansion
       (alias fanout, removed-buffer tracking) and calls
       ``compile_with_fusion`` with the fully-expanded spec lists.
    4. ``ExternalTemplateBuffer.call_kernel`` calls ``emit_call_site``.
    5. ``ExternalTemplateBuffer.emit_kernel_override`` calls
       ``emit_kernel_to_header``.
    """

    def __init__(
        self,
        kernel: "Kernel",
        bound_kernel: "BoundKernel",
        named_input_nodes: dict[str, IRNode],
        constant_args: dict[str, object],
        output_buf_to_param: dict[str, tuple[str | None, list[tuple[type, int]]]],
        metadata: TemplateKernelMetadata | None,
        autotune_args: tuple[object, ...] | None = None,
    ) -> None:
        self._kernel = kernel
        self._bound_kernel = bound_kernel
        self._named_input_nodes = named_input_nodes
        self._constant_args = constant_args
        # Shared mutable dict: populated by build_multi_outputs callbacks in
        # lower_helion_kernel; metadata is set once it's complete.
        self._output_buf_to_param = output_buf_to_param
        self._metadata = metadata
        self._autotune_args = autotune_args

        # Active fusion specs — set at the start of compile_with_fusion and
        # read by the AST transform callbacks during _generate_triton_ast.
        self._active_epilogue_specs: dict[str, ResolvedEpilogueSpec] = {}
        self._active_prologue_specs: dict[str, ResolvedPrologueSpec] = {}

    # ------------------------------------------------------------------ #
    # ExternalKernelBackend protocol                                       #
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> TemplateKernelMetadata:
        assert self._metadata is not None, (
            "HelionKernelBackend.get_metadata() called before metadata was set"
        )
        return self._metadata

    def get_param_alias_map(self) -> dict[str, list[str]]:
        """Return ``{buf_name: [param_name, ...]}``.

        Inductor uses this to fan out prologue specs to every parameter name
        under which the same input buffer appears (handles ``k_add(x, x)``).
        """
        result: dict[str, list[str]] = {}
        for param_name, inp in self._named_input_nodes.items():
            result.setdefault(inp.get_name(), []).append(param_name)  # type: ignore[union-attr]
        return result

    def compile_with_fusion(
        self,
        epilogue_specs: list[ResolvedEpilogueSpec],
        prologue_specs: dict[str, ResolvedPrologueSpec],
        extra_params: list[tuple[str, str]],
    ) -> str:
        """Generate complete Triton kernel source with fusion inlined.

        Called by ``ExternalTemplateBuffer.codegen_with_fusion`` after
        Inductor has applied spec expansion.  The specs are fully resolved
        (Triton expression strings); this method only needs to:

        1. Store the specs so the AST transform callbacks can read them.
        2. Autotune (if needed) before the fused AST is generated.
        3. Generate the Triton AST with ``store_transform`` /
           ``load_transform`` callbacks active.
        4. Inject extra fusion parameters into the AST.
        5. Return the serialised source string.

        ``prologue_specs`` is a ``{kernel_param_name: spec}`` dict that has
        already had alias expansion applied by
        ``ExternalTemplateBuffer.codegen_with_fusion``: when the same input
        buffer appears under multiple parameter names (e.g. ``k_add(x, x)``),
        each parameter name gets its own entry so that both load sites get
        the fused expression inlined.
        """
        # Store active specs — the _codegen_{epilogue,prologue}_fusion
        # callbacks read these during generate_ast.
        self._active_epilogue_specs = {s.kernel_output_param: s for s in epilogue_specs}
        # prologue_specs is already keyed by param name with alias expansion done.
        self._active_prologue_specs = dict(prologue_specs)

        # Autotune must fire *before* the fused AST is generated so that the
        # autotuned (unfused) code does not contain fusion patterns.
        if (epilogue_specs or prologue_specs) and self._autotune_args and self._bound_kernel:
            self._bound_kernel.ensure_config_exists(self._autotune_args)

        root = self._generate_triton_ast()
        if root is None:
            return ""

        # Inject extra params (epilogue inputs / redirected outputs) into the
        # inner device function, the host wrapper, and the launcher call.
        if extra_params:
            epilogue_renames = {
                s.kernel_output_param: s.redirect_param
                for s in epilogue_specs
                if s.redirect_param is not None
            }
            self._inject_fusion_params(root, extra_params, epilogue_renames)

        return self._ast_to_source(root)

    def emit_call_site(
        self,
        kernel_name: str,
        output_name: str,
        wrapper: "PythonWrapperCodegen",
        prologue_specs: "dict[str, ResolvedPrologueSpec]",
        epilogue_extra_params: "list[tuple[str, str]]",
        removed_buffers: "OrderedSet[str]",
    ) -> None:
        """Emit the kernel call and MultiOutput extraction into the wrapper."""
        reinterp_count = 0

        def get_input_expr(arg_name: str, inp: IRNode) -> str:
            nonlocal reinterp_count
            buf_name = inp.get_name()  # type: ignore[union-attr]
            pro_spec = prologue_specs.get(arg_name)
            source_buf = pro_spec.source_buf if pro_spec is not None else None

            if source_buf is not None:
                # This input is prologue-fused: use the source buffer, but
                # preserve any ReinterpretView strides / offsets.
                if isinstance(inp, ReinterpretView):
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
            for name, inp in self._named_input_nodes.items()
        }

        all_args: dict[str, object] = {n: _CodeExpr(v) for n, v in arg_inputs.items()}
        for n, v in self._constant_args.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        sig = self._kernel.signature.parameters
        args = [
            repr(all_args[n]) if n in all_args else repr(p.default)
            for n, p in sig.items()
            if n in all_args or p.default is not p.empty
        ]

        # Append epilogue extra parameters (fused outputs and inputs).
        args.extend(buf_name for _, buf_name in epilogue_extra_params)
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Emit MultiOutput extraction assignments.  MultiOutput nodes are
        # marked as run by codegen_with_fusion, so their separate codegen is
        # suppressed; we must emit the assignments here instead.
        for mo_name, (_param, indices) in sorted(self._output_buf_to_param.items()):
            if mo_name not in removed_buffers:
                idx_str = output_name
                for _, idx in indices:
                    idx_str = f"{idx_str}[{idx}]"
                wrapper.writeline(f"{mo_name} = {idx_str}")

    def emit_kernel_to_header(
        self,
        wrapper: "PythonWrapperCodegen",
        src_code: str,
        kernel_name: str,
        node_schedule: "Sequence[BaseSchedulerNode | object]",
        kernel_path: str,
        get_kernel_metadata: "Callable[[Sequence[BaseSchedulerNode | object], PythonWrapperCodegen], tuple[str, str]]",
    ) -> bool:
        """Add imports and write kernel source to the wrapper header."""
        required = ("triton", "tl", "_default_launcher")
        conditional = ("libdevice", "tl_math", "triton_helpers", "helion", "hl")
        for name in (*required, *(n for n in conditional if f"{n}." in src_code)):
            wrapper.add_import_once(library_imports[name])

        # Add imports for globals captured by the kernel (e.g. user modules).
        if self._bound_kernel.host_function is not None:
            for imp in self._bound_kernel.host_function.global_imports.values():
                wrapper.add_import_once(imp.codegen())

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        # Skip import lines at the top of the generated source (they are
        # emitted via add_import_once above) and write the rest to the header.
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

    # ------------------------------------------------------------------ #
    # Private Helion-specific helpers                                      #
    # ------------------------------------------------------------------ #

    def _generate_triton_ast(self) -> ast.Module | None:
        """Generate and rename the Triton kernel AST.

        Activates ``store_transform`` / ``load_transform`` callbacks when
        active fusion specs are present so that ``hl.store`` / ``hl.load``
        sites inline fused expressions directly.
        """
        if not self._bound_kernel:
            return None
        # Config must be available before AST generation.
        if self._autotune_args and not (
            self._active_epilogue_specs or self._active_prologue_specs
        ):
            # Fusion path calls ensure_config_exists at the top of
            # compile_with_fusion; no-fusion path does it here.
            self._bound_kernel.ensure_config_exists(self._autotune_args)

        cfg = self._bound_kernel._config
        assert cfg is not None, "Config should be set after ensure_config_exists"
        host_fn = self._kernel.name
        inner_fn = f"_helion_{host_fn}"
        inner_fn_placeholder = f"{inner_fn}_{Placeholder.KERNEL_NAME}"

        with self._bound_kernel.env:
            host_function = self._bound_kernel.host_function
            assert host_function is not None, "BoundKernel must have a host_function"
            root = generate_ast(
                host_function,
                cfg,
                emit_repro_caller=False,
                store_transform=self._codegen_epilogue_fusion
                if self._active_epilogue_specs
                else None,
                load_transform=self._codegen_prologue_fusion
                if self._active_prologue_specs
                else None,
            )

        assert isinstance(root, ast.Module)

        # Collect module-level variable names for uniquification
        # (e.g. constexpr assignments like ``_BLOCK_SIZE_0 = tl.constexpr(32)``).
        module_level_vars: dict[str, str] = {}
        for node in root.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_level_vars[target.id] = (
                            f"{target.id}_{Placeholder.KERNEL_NAME}"
                        )

        # Rename functions, module-level vars, and all references to them.
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
        return get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )

    def _inject_fusion_params(
        self,
        root: ast.Module,
        extra_params: list[tuple[str, str]],
        epilogue_renames: dict[str, str],
    ) -> None:
        """Inject extra fusion parameters into the inner function, host
        function, and launcher call in the generated AST.

        Helion always generates the inner (device) function first, then the
        host wrapper — this invariant is checked below.
        """
        funcs = [n for n in ast.iter_child_nodes(root) if isinstance(n, ast.FunctionDef)]
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

        extra_param_names = [p for p, _ in extra_params]
        for name in extra_param_names:
            inner_func.args.args.append(ast.arg(arg=name))
            host_func.args.args.append(ast.arg(arg=name))
            if launcher_call is not None:
                launcher_call.args.append(ast.Name(id=name, ctx=ast.Load()))

        # Apply redirect renames so tensor_descriptor params get the right name.
        if epilogue_renames:
            func_params = {arg.arg for arg in inner_func.args.args}
            for orig_param, new_param in epilogue_renames.items():
                if orig_param not in func_params:
                    raise RuntimeError(
                        f"Epilogue redirect expected '{orig_param}' to be a "
                        f"parameter of the Triton inner function, but found "
                        f"only: {sorted(func_params)}.  This indicates a "
                        f"mismatch between the kernel's output param naming "
                        f"and the epilogue spec's kernel_output_param."
                    )
                ast_rename(inner_func, {orig_param: new_param})

    def _codegen_epilogue_fusion(
        self,
        state: "CodegenState",
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        """Apply resolved epilogue spec at an ``hl.store`` site.

        Called from ``memory_ops.py`` via the ``store_transform`` callback
        passed to ``generate_ast``.  Resolves Helion-specific indexing context
        (offset, mask, per-dimension index expressions) and delegates the
        generic expression-application logic to
        ``TritonTemplateBuffer.apply_resolved_epilogue_at_store``.
        """
        assert self._active_epilogue_specs
        param_name = state.device_function.tensor_arg(tensor).name
        spec = self._active_epilogue_specs.get(param_name)
        if spec is None:
            return value

        # Unique per-epilogue name avoids Triton type conflicts across branches.
        epi_idx = list(self._active_epilogue_specs.keys()).index(param_name)
        kernel_val_name = f"_kernel_val_{epi_idx}"

        # Resolve Helion-specific indexing context for this store site.
        indexing = SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
        offset_str = ast.unparse(indexing.index_expr)
        mask_str = ast.unparse(indexing.mask_expr) if indexing.has_mask() else None

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
        """Apply resolved prologue spec at an ``hl.load`` site.

        Called from ``memory_ops.py`` via the ``load_transform`` callback
        passed to ``generate_ast``.  Substitutes the ``_load_val`` placeholder
        in the pre-traced fused expression with the actual load AST.
        """
        assert self._active_prologue_specs
        param_name = state.device_function.tensor_arg(tensor).name
        spec = self._active_prologue_specs.get(param_name)
        if spec is None:
            return value
        return TritonTemplateBuffer.apply_resolved_prologue_at_load(spec, value)  # type: ignore[return-value]


def _flatten_return_ast(
    ast_node: ast.expr | None,
    structured: object,
) -> list[ast.expr | None]:
    """Get the per-leaf AST nodes in DFS order matching build_multi_outputs traversal.

    Walks ``structured`` in the same order as ``build_multi_outputs`` to produce
    a flat list mapping ``leaf_idx`` → the corresponding AST node from the
    kernel's return statement.  Used to extract kernel parameter names
    (``ast.Name`` nodes) and detect symbolic (non-constant) non-tensor returns.
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
    """Lower a Helion kernel HOP to an ``ExternalTemplateBuffer``.

    Creates a ``HelionKernelBackend`` (pure Helion logic) and wraps it in
    Inductor's generic ``ExternalTemplateBuffer`` IR node.  Inductor then
    schedules the buffer and calls into the backend through the
    ``ExternalKernelBackend`` protocol for fusion planning and codegen.
    """
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Realize inputs: convert TensorBox → buffer / ReinterpretView.
    # Use TritonTemplateBuffer.realize_template_input to preserve MultiOutput
    # layouts (ExternKernel.realize_input would lose non-contiguous strides).
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = TritonTemplateBuffer.realize_template_input(tb)

    # Build fake tensors for kernel binding (sympy exprs → concrete ints).
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

    # Derive output structure from the bound kernel using inductor-time layouts.
    flat_leaves, tree_spec, return_ast = _get_flat_output(bound.host_function)
    example_outputs = [leaf for leaf in flat_leaves if isinstance(leaf, torch.Tensor)]

    dev = (
        example_outputs[0].device
        if example_outputs
        else inputs[0].get_device()
        if inputs
        else torch.device("cuda")
    )
    assert dev is not None

    # Shared mutable dict: callbacks below populate it during build_multi_outputs;
    # metadata is set once it is complete.
    output_buf_to_param: dict[str, tuple[str | None, list[tuple[type, int]]]] = {}

    # Resolve mutated inputs to IRNode references.
    mutated_inputs_irnodes = [
        realized[n] for n in (mutated_inputs_list or []) if n in realized
    ] or None

    # Create the backend (pure Helion, no Inductor IR inheritance).
    backend = HelionKernelBackend(
        kernel=kernel,
        bound_kernel=bound,
        named_input_nodes=dict(zip(realized.keys(), realized.values())),
        constant_args=constant_args,
        output_buf_to_param=output_buf_to_param,
        metadata=None,  # set below, after build_multi_outputs fills output_buf_to_param
        autotune_args=tuple(fake_tensors),
    )

    # Create the generic Inductor IR node that wraps the backend.
    buf = ExternalTemplateBuffer(
        layout=MultiOutputLayout(device=dev),
        inputs=inputs,
        backend=backend,
        mutated_inputs=mutated_inputs_irnodes,
        allowed_prologue_inps=OrderedSet(
            inp.get_name()
            for inp in inputs  # type: ignore[union-attr]
        ),
    )

    for inp in mutated_inputs_irnodes or []:
        if hasattr(inp, "get_name"):
            V.graph.never_reuse_buffers.add(inp.get_name())

    if not example_outputs:
        backend._metadata = TemplateKernelMetadata(
            all_inputs={inp.get_name(): p for p, inp in backend._named_input_nodes.items()},  # type: ignore[union-attr]
            fusable_outputs={},
            all_output_names=set(),
            mutated_input_names=mutated_inputs_list or [],
        )
        return ()

    # Reconstruct structured output and create MultiOutput nodes.
    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    # Flatten return_ast to index by leaf_idx (same traversal as build_multi_outputs).
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
        output_buf_to_param[mo_name] = (
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

    # Compute fusable_outputs: param known + no symbolic returns + shape matches an input.
    input_shapes = {tuple(inp.get_size()) for inp in inputs}  # type: ignore[union-attr]
    fusable_outputs = {
        mo_name: param
        for mo_name, (param, _) in output_buf_to_param.items()
        if param is not None
        and not has_symbolic_returns_flag[0]
        and (not input_shapes or output_sizes.get(mo_name, ()) in input_shapes)
    }

    # Now that output_buf_to_param is fully populated, set metadata on the backend.
    backend._metadata = TemplateKernelMetadata(
        all_inputs={inp.get_name(): p for p, inp in backend._named_input_nodes.items()},  # type: ignore[union-attr]
        fusable_outputs=fusable_outputs,
        all_output_names=set(output_buf_to_param),
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
