#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <utility>
#include <functional>
#include <numeric>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__global__ void matmul(const float *N_float, const float *diag_number_float, const float *A, const float *offsets, const float *B, float *C){
    int64_t N = static_cast<int64_t>(*N_float);
    int64_t diag_number = static_cast<int64_t>(*diag_number_float);

    int row = blockIdx.y * 32 + (threadIdx.x % 32);
    int col = blockIdx.x * 32 + (threadIdx.x / 32);

    if (row < N && col < N){
        for (int i = 0; i < diag_number; i++){
            int offset = static_cast<int>(offsets[i]);
            bool isOffsetPositive = (offset >= 0);
            if (isOffsetPositive){
                if (row < N - offset){
                    C[row * N + col] += A[row+i*N+offset] * B[row*N+col+offset*N];
                }
            } else {
                if (row < N + offset) {
                    C[row*N+col-offset*N] += A[row+i*N] * B[row*N+col];
                }
            }
        }
    }
}

template <ffi:: DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer){
    auto dims = buffer.dimensions();
    if (dims.size() == 0){
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error MatmulDIAImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> N,
    ffi::Buffer<ffi::DataType::F32> diag_number,
    ffi::Buffer<ffi::DataType::F32> A,
    ffi::Buffer<ffi::DataType::F32> offsets,
    ffi::Buffer<ffi::DataType::F32> B,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> C
){
    const int block_size = 32;

    auto [totalSize, lastDim] = GetDims(B);

    dim3 block(block_size * block_size);
    dim3 grid(lastDim/block_size, lastDim/block_size);
    
    matmul<<<grid, block, 0, stream>>>(N.typed_data(), diag_number.typed_data(), A.typed_data(), offsets.typed_data(), B.typed_data(), C -> typed_data());

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MatmulDIA, 
    MatmulDIAImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
);