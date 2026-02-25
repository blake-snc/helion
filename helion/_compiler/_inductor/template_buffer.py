from __future__ import annotations

import ast
import contextlib
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
from torch._inductor.ir import ExternalTritonTemplateBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import KernelSource
from torch._inductor.ir import Layout
from torch._inductor.ir import OutputSpec
from torch._inductor.ir import EpilogueSpec
from torch._inductor.ir import PrologueSpec
from torch._inductor.ir import TensorBox
from torch._inductor.ir import TemplateKernelMetadata
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.lowering import register_lowering
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


class HelionTritonTemplateBuffer(ExternalTritonTemplateBuffer):
    """Helion's concrete ``ExternalTritonTemplateBuffer`` implementation.

    Combines the Inductor IR node with Helion's kernel codegen logic by
    implementing ``describe()`` and ``compile()`` directly — no separate
    backend object needed.

    Lifecycle
    ---------
    1. ``lower_helion_kernel`` calls ``HelionTritonTemplateBuffer.from_kernel``
       which builds the IR node and returns ``(buf, outputs)``.
    2. The caller sets ``buf._metadata`` after computing ``fusable_outputs``
       (which needs the output-info collected during ``from_kernel``).
    3. Inductor's scheduler queries ``describe()`` (returns ``_metadata``) to
       plan fusion.
    4. ``codegen_with_fusion`` (inherited) applies spec expansion and calls
       ``compile()``.  The returned ``KernelSource`` (source + imports +
       call_args + call_preamble) is cached on ``self._artifact``.
    5. ``call_kernel`` (inherited) emits the call using the cached artifact —
       no further involvement of this class.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs: "Sequence[IRNode]",
        *,
        kernel: "Kernel",
        bound_kernel: "BoundKernel",
        constant_args: dict[str, object],
        autotune_args: tuple[object, ...] | None = None,
        mutated_inputs: "Optional[Iterable[IRNode]]" = None,
        allowed_prologue_inps: "Optional[OrderedSet[str]]" = None,
    ) -> None:
        self._kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args = constant_args
        self._autotune_args = autotune_args
        self._metadata: TemplateKernelMetadata | None = None

        super().__init__(
            layout=layout,
            inputs=inputs,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )

    # ------------------------------------------------------------------ #
    # ExternalTritonTemplateBuffer abstract methods                        #
    # ------------------------------------------------------------------ #

    def describe(self) -> TemplateKernelMetadata:
        assert self._metadata is not None, (
            "HelionTritonTemplateBuffer.describe() called before metadata was set"
        )
        return self._metadata

    def compile(
        self,
        epilogue_specs: list[EpilogueSpec],
        prologue_specs: dict[str, PrologueSpec],
        extra_params: list[tuple[str, str]],
    ) -> KernelSource:
        """Generate source, imports, and the full kernel call-site in one shot.

        Called once by the inherited ``codegen_with_fusion`` with the complete,
        alias-expanded fusion context.  Returns source, imports, call_args, and
        call_preamble so ``call_kernel`` needs no further backend involvement.
        """
        # 1. Autotune before AST generation (must fire before fused AST is generated
        #    so that the autotuned (unfused) code does not contain fusion patterns).
        if (epilogue_specs or prologue_specs) and self._autotune_args and self._bound_kernel:
            self._bound_kernel.ensure_config_exists(self._autotune_args)

        # 3. Generate Triton AST with store/load transform callbacks active
        root = self._generate_triton_ast()
        if root is None:
            return KernelSource(source="", imports=[])

        # 4. Inject extra fusion params (epilogue inputs / redirected outputs)
        if extra_params:
            epilogue_renames = {
                s.kernel_output_param: s.redirect_param
                for s in epilogue_specs
                if s.redirect_param is not None
            }
            self._inject_fusion_params(root, extra_params, epilogue_renames)

        # 5. Serialize to source
        source = self._ast_to_source(root)

        # 6. Compute imports from source
        required = ("triton", "tl", "_default_launcher")
        conditional = ("libdevice", "tl_math", "triton_helpers", "helion", "hl")
        imports = [
            library_imports[n]
            for n in (*required, *(n for n in conditional if f"{n}." in source))
        ]
        if self._bound_kernel.host_function is not None:
            imports.extend(
                imp.codegen()
                for imp in self._bound_kernel.host_function.global_imports.values()
            )

        # 7. Compute call args and preamble
        call_order, constant_repr = self._call_order_and_constant_repr()
        call_preamble, call_args = self._resolve_call_args(
            call_order, constant_repr, prologue_specs, extra_params
        )

        return KernelSource(
            source=source,
            imports=imports,
            call_args=call_args,
            call_preamble=call_preamble,
        )

    # ------------------------------------------------------------------ #
    # Metadata helpers (called by lower_helion_kernel after from_kernel)  #
    # ------------------------------------------------------------------ #

    def _call_order_and_constant_repr(self) -> tuple[list[str], dict[str, str]]:
        """Compute the kernel call order and pre-repr'd non-tensor args.

        ``call_order`` lists every parameter name in signature order.
        ``constant_repr`` maps non-tensor param names to their ``repr()``-ready
        strings (scalars, defaults, and rebuilt container args) so the inherited
        ``call_kernel`` can emit them without calling back into this class.
        """
        # Both tensor inputs AND constant args must be combined before
        # _rebuild_container_args so it can pop 'param.0', 'param.1' etc.
        all_args: dict[str, object] = {
            n: _CodeExpr(inp.get_name())  # type: ignore[union-attr]
            for n, inp in self._named_inputs.items()
        }
        for n, v in self._constant_args.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        tensor_flat_params = frozenset(self._named_inputs.keys())
        sig = self._kernel.signature.parameters
        order: list[str] = []
        const_repr: dict[str, str] = {}
        for n, p in sig.items():
            if n in all_args:
                order.append(n)
                if n not in tensor_flat_params:
                    const_repr[n] = repr(all_args[n])
            elif p.default is not p.empty:
                order.append(n)
                const_repr[n] = repr(p.default)
        return order, const_repr

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
            self._epilogue_specs or self._prologue_specs
        ):
            # Fusion path calls ensure_config_exists at the top of compile();
            # no-fusion path does it here.
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
                if self._epilogue_specs
                else None,
                load_transform=self._codegen_prologue_fusion
                if self._prologue_specs
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
        assert self._epilogue_specs
        param_name = state.device_function.tensor_arg(tensor).name
        spec = self._epilogue_specs.get(param_name)
        if spec is None:
            return value

        # Unique per-epilogue name avoids Triton type conflicts across branches.
        epi_idx = list(self._epilogue_specs.keys()).index(param_name)
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
        assert self._prologue_specs
        param_name = state.device_function.tensor_arg(tensor).name
        spec = self._prologue_specs.get(param_name)
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
    """Lower a Helion kernel HOP to a ``HelionTritonTemplateBuffer``.

    Calls ``HelionTritonTemplateBuffer.from_kernel`` which builds the Inductor
    IR node and returns ``(buf, outputs)``.  After the multi-output structure
    is known, sets ``buf._metadata`` so that ``describe()`` can serve it.
    Inductor then schedules the buffer and calls ``describe()`` / ``compile()``
    for fusion planning and codegen.
    """
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Realize inputs: convert TensorBox → buffer / ReinterpretView.
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = HelionTritonTemplateBuffer.realize_template_input(tb)

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

    # Derive output structure from the bound kernel using inductor-time layouts.
    flat_leaves, tree_spec, return_ast = _get_flat_output(bound.host_function)

    if not flat_leaves:
        # No outputs — from_kernel still creates the buffer for mutations.
        buf, _ = HelionTritonTemplateBuffer.from_kernel(
            realized_inputs=realized,
            structured_outputs=None,
            mutated_input_names=mutated_inputs_list or [],
            direct_aliases={},
            kernel=kernel,
            bound_kernel=bound,
            constant_args=constant_args,
            autotune_args=tuple(fake_tensors),
        )
        buf._metadata = TemplateKernelMetadata(
            all_inputs={inp.get_name(): p for p, inp in buf._named_inputs.items()},  # type: ignore[union-attr]
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

    output_info: dict[str, tuple[str | None, list[tuple[type, int]]]] = {}
    output_sizes: dict[str, tuple[object, ...]] = {}
    has_symbolic_returns_flag = [False]

    def on_tensor_leaf(
        mo_name: str,
        mo: "MultiOutput",
        indices: list[tuple[type, int]],
        leaf_idx: int,
    ) -> None:
        ast_node = flat_ast[leaf_idx] if leaf_idx < len(flat_ast) else None
        output_info[mo_name] = (
            ast_node.id if isinstance(ast_node, ast.Name) else None,
            indices,
        )
        output_sizes[mo_name] = tuple(mo.get_size())

    def on_non_tensor_leaf(leaf_idx: int) -> None:
        ast_node = flat_ast[leaf_idx] if leaf_idx < len(flat_ast) else None
        if ast_node is not None and not isinstance(ast_node, ast.Constant):
            has_symbolic_returns_flag[0] = True

    buf, result = HelionTritonTemplateBuffer.from_kernel(
        realized_inputs=realized,
        structured_outputs=structured,
        mutated_input_names=mutated_inputs_list or [],
        direct_aliases={
            i: realized[name]
            for i, name in cast(
                "dict[int, str]", output_spec.get("direct_aliases", {})
            ).items()
            if name in realized
        },
        on_tensor_leaf=on_tensor_leaf,
        on_non_tensor_leaf=on_non_tensor_leaf,
        kernel=kernel,
        bound_kernel=bound,
        constant_args=constant_args,
        autotune_args=tuple(fake_tensors),
    )

    # Compute fusable_outputs: param known + no symbolic returns + shape matches an input.
    input_shapes = {tuple(inp.get_size()) for inp in realized.values()}  # type: ignore[union-attr]
    fusable_outputs = {
        mo_name: param
        for mo_name, (param, _) in output_info.items()
        if param is not None
        and not has_symbolic_returns_flag[0]
        and (not input_shapes or output_sizes.get(mo_name, ()) in input_shapes)
    }

    buf._metadata = TemplateKernelMetadata(
        all_inputs={inp.get_name(): p for p, inp in buf._named_inputs.items()},  # type: ignore[union-attr]
        fusable_outputs=fusable_outputs,
        all_output_names=set(output_info),
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
