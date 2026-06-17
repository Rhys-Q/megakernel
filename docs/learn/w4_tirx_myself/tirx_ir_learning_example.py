"""A compact TIRX/sTIR IR learning example.

This file is for reading and printing IR shape, not for performance.
The single example is marked ``s_tir=True`` so SBlock/SBlockRealize is legal,
while it still includes TIRX scope/layout/tile-primitive constructs for learning.

Covered pieces: PrimFunc, PrimExpr, Stmt, Buffer, IterVar, BufferRegion,
ExecScope, Layout, and TilePrimitiveCall.
"""

from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TileLayout, laneid


def gmem_layout():
    """Simple global-memory logical layout."""
    return TileLayout(S[16])


def smem_layout():
    """A small shared-memory layout sharded by lane id."""
    return TileLayout(S[16 : 1 @ laneid])


# from tvm.script import tirx as Tx

@Tx.prim_func(s_tir=True)
def tirx_ir_learning(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
    """One legal learning example covering the main TIRX/sTIR IR objects."""
    A = Tx.match_buffer(A_ptr, (16,), "float32", scope="global", layout=gmem_layout())
    B = Tx.match_buffer(B_ptr, (16,), "float32", scope="global", layout=gmem_layout())

    with Tx.kernel():
        bx = Tx.cta_id([1])

        with Tx.cta():
            Sbuf = Tx.alloc_buffer(
                (16,),
                "float32",
                scope="shared",
                layout=smem_layout(),
            )

            Tx.copy(Sbuf[0:16], A[0:16])

            for i in Tx.serial(16):
                with Tx.sblock("add_one"):
                    vi = Tx.axis.spatial(16, i)
                    Tx.reads(Sbuf[vi])
                    Tx.writes(B[vi])
                    B[vi] = Sbuf[vi] + Tx.float32(1)


def _line(indent, text):
    print("  " * indent + text)


def _short(obj):
    return str(obj).replace("\n", " ")


def _var_name(var):
    return getattr(var, "name_hint", getattr(var, "name", str(var)))


def dump_expr(expr, indent=0, label="expr"):
    """Print a compact PrimExpr tree for the nodes used in this example."""
    node = type(expr).__name__
    dtype = getattr(expr, "dtype", None)
    dtype_suffix = f", dtype={dtype}" if dtype is not None else ""

    if hasattr(expr, "a") and hasattr(expr, "b"):
        _line(indent, f"{label}: {node}{dtype_suffix}")
        dump_expr(expr.a, indent + 1, "a")
        dump_expr(expr.b, indent + 1, "b")
    elif node == "BufferLoad":
        _line(indent, f"{label}: BufferLoad{dtype_suffix}")
        _line(indent + 1, f"buffer: {expr.buffer.name}")
        _line(indent + 1, f"indices: {[_short(i) for i in expr.indices]}")
    elif node in ("IntImm", "FloatImm"):
        _line(indent, f"{label}: {node}(value={expr.value}, dtype={expr.dtype})")
    elif node in ("Var", "SizeVar"):
        _line(indent, f"{label}: {node}(name={_var_name(expr)}, dtype={expr.dtype})")
    else:
        _line(indent, f"{label}: {node}({_short(expr)}){dtype_suffix}")


def dump_stmt(stmt, indent=0):
    """Print a compact Stmt tree for the nodes used in this example."""
    node = type(stmt).__name__

    if node == "SBlockRealize":
        _line(indent, "SBlockRealize")
        _line(indent + 1, f"iter_values: {[_short(v) for v in stmt.iter_values]}")
        _line(indent + 1, f"predicate: {_short(stmt.predicate)}")
        dump_stmt(stmt.block, indent + 1)
    elif node == "SBlock":
        _line(indent, f"SBlock(name={stmt.name_hint})")
        _line(indent + 1, f"iter_vars: {[_short(v) for v in stmt.iter_vars]}")
        _line(indent + 1, f"reads: {[_short(r) for r in stmt.reads]}")
        _line(indent + 1, f"writes: {[_short(w) for w in stmt.writes]}")
        dump_stmt(stmt.body, indent + 1)
    elif node == "ExecScopeStmt":
        _line(indent, f"ExecScopeStmt(kind={stmt.exec_scope.name})")
        dump_stmt(stmt.body, indent + 1)
    elif node == "SeqStmt":
        _line(indent, f"SeqStmt(len={len(stmt.seq)})")
        for i, child in enumerate(stmt.seq):
            _line(indent + 1, f"[{i}]")
            dump_stmt(child, indent + 2)
    elif node == "AllocBuffer":
        buf = stmt.buffer
        _line(
            indent,
            "AllocBuffer("
            f"name={buf.name}, shape={[_short(s) for s in buf.shape]}, "
            f"dtype={buf.dtype}, scope={buf.scope()}, layout={_short(buf.layout)}"
            ")",
        )
    elif node == "TilePrimitiveCall":
        _line(indent, f"TilePrimitiveCall(op={stmt.op.name})")
        _line(indent + 1, f"args: {[_short(arg) for arg in stmt.args]}")
    elif node == "For":
        _line(
            indent,
            f"For(loop_var={_var_name(stmt.loop_var)}, min={stmt.min}, "
            f"extent={stmt.extent}, kind={stmt.kind})",
        )
        dump_stmt(stmt.body, indent + 1)
    elif node == "BufferStore":
        _line(indent, f"BufferStore(buffer={stmt.buffer.name})")
        _line(indent + 1, f"indices: {[_short(i) for i in stmt.indices]}")
        dump_expr(stmt.value, indent + 1, "value")
    else:
        _line(indent, f"{node}: {_short(stmt)}")


def dump_primfunc(func):
    """Print the PrimFunc and its main TIRX/sTIR IR tree."""
    _line(0, f"PrimFunc(name={func.attrs.get('global_symbol', '<anonymous>')})")
    _line(1, f"attrs: {func.attrs}")
    _line(1, f"params: {[_short(p) for p in func.params]}")
    _line(1, "buffer_map:")
    for param, buffer in func.buffer_map.items():
        _line(
            2,
            f"{param} -> Buffer(name={buffer.name}, shape={[_short(s) for s in buffer.shape]}, "
            f"dtype={buffer.dtype}, scope={buffer.scope()}, layout={_short(buffer.layout)})",
        )
    _line(1, "body:")
    dump_stmt(func.body, 2)


def main():
    print("=== TVMScript ===")
    print(tirx_ir_learning.script())
    print()
    print("=== Compact TIRX IR tree ===")
    dump_primfunc(tirx_ir_learning)


if __name__ == "__main__":
    main()
