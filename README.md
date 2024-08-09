## Extending JAX with custom CUDA kernels

This repo describes how to extend the capabilities of JAX following the latest contributions to their documentation. Here we are writing our own custom CUDA kernels and call them in Python as compiled libraries through a FFI ("Foreign Function Interface"). Our demo illustrates an example of the SparseDIA matmul (not the finished version) and I will add on more features soon.

### Description of the files
- `dia_cuda.cu`: CUDA kernel of the SparseDIA matmul
- `dia_jax.py`: Python code that displays how to use the CUDA kernels.
- `CMakeLists.txt`: this will compile our kernels in usable libraries in Python

### Set up
Run these commands in the root directory:
```
!cmake -DCMAKE_BUILD_TYPE=Release -B ffi/_build ffi
!cmake --build ffi/_build
!cmake --install ffi/_build
```

and then run the Python code normally.
