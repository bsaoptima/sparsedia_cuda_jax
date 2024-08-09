
import ctypes
from pathlib import Path
import jax.extend as jex
import jax
import jax.numpy as jnp
import numpy as np

path = next(Path("/home/bagrici/cuda_dia/ffi/_build").glob("libdia_cuda*"))
dia_cuda = ctypes.cdll.LoadLibrary(path)
jex.ffi.register_ffi_target(
    "dia_cuda",
    jex.ffi.pycapsule(dia_cuda.MatmulDIA),
    platform="CUDA"
)

def matmul(N, diag_number, A, offsets, B):
    out_type = jax.ShapeDtypeStruct(B.shape, B.dtype)

    return jex.ffi.ffi_call(
        "dia_cuda",
        out_type,
        N,
        diag_number,
        A,
        offsets,
        B,
        vectorized=True
    )


N = 64
diag_number = 2
A = jnp.arange(N*diag_number).reshape(diag_number, N)
offsets = jnp.array([1,2])
B = jnp.arange(N**2).reshape(N,N)

out= matmul(np.float32(N), np.float32(diag_number), np.float32(A), np.float32(offsets), np.float32(B))
print(out)
print(jnp.sum(out))
