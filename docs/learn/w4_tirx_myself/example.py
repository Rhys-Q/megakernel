from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def add_one(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")) -> None:
    for i in range(16):
        B[i] = A[i] + T.float32(1.0)
print(type(add_one))