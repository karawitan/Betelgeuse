#pragma once

// Minimal CUDA -> HIP runtime shim for the src/ MoE streaming code.
//
// The llama target is compiled by the host C++ compiler in both CUDA and HIP
// builds, so ggml/src/ggml-cuda/vendors/hip.h is not reachable from here.
// This header exposes only the API subset used by moe_pager.cpp,
// moe_doorbell.cu and moe_predictor.cu under HIP.

#if defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define cudaStream_t                  hipStream_t
#define cudaStreamCreateWithFlags     hipStreamCreateWithFlags
#define cudaStreamNonBlocking         hipStreamNonBlocking
#define cudaStreamSynchronize         hipStreamSynchronize
#define cudaStreamDestroy             hipStreamDestroy
#define cudaStreamIsCapturing         hipStreamIsCapturing
#define cudaStreamCaptureStatus       hipStreamCaptureStatus
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusNone   hipStreamCaptureStatusNone

#define cudaEvent_t                   hipEvent_t
#define cudaEventCreate               hipEventCreate
#define cudaEventCreateWithFlags      hipEventCreateWithFlags
#define cudaEventDisableTiming        hipEventDisableTiming
#define cudaEventRecord               hipEventRecord
#define cudaEventElapsedTime          hipEventElapsedTime
#define cudaEventQuery                hipEventQuery
#define cudaEventSynchronize          hipEventSynchronize
#define cudaEventDestroy              hipEventDestroy

#define cudaError_t                   hipError_t
#define cudaSuccess                   hipSuccess
#define cudaErrorNotReady             hipErrorNotReady
#define cudaGetErrorString            hipGetErrorString
#define cudaMemGetInfo                hipMemGetInfo

#define cudaMalloc                    hipMalloc
#define cudaFree                      hipFree
#define cudaMemset                    hipMemset
#define cudaMemcpy                    hipMemcpy
#define cudaMemcpyAsync               hipMemcpyAsync
#define cudaMemsetAsync               hipMemsetAsync
#define cudaMemcpyHostToDevice        hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost        hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice      hipMemcpyDeviceToDevice

#define cudaStreamWaitEvent           hipStreamWaitEvent

#define cudaHostAlloc                 hipHostMalloc
#define cudaHostAllocDefault          hipHostMallocDefault
#define cudaHostAllocMapped           hipHostMallocMapped
#define cudaHostAllocPortable         hipHostMallocPortable
#define cudaHostAllocWriteCombined    hipHostMallocWriteCombined
#define cudaHostGetDevicePointer      hipHostGetDevicePointer
#define cudaHostRegister              hipHostRegister
#define cudaHostRegisterDefault       hipHostRegisterDefault
#define cudaHostUnregister            hipHostUnregister
#define cudaFreeHost                  hipHostFree

#define cudaSetDevice                 hipSetDevice
#define cudaGetDevice                 hipGetDevice

#else

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#endif
