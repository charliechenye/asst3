#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
__global__ void upsweep_kernel(const int two_d, const int two_dplus1, int* input, int* output, const int N){
    unsigned int idx {blockIdx.x * blockDim.x + threadIdx.x};
    unsigned int k {two_dplus1 * idx};
    if (k < N){
        output[k + two_dplus1 - 1] = input[k + two_d - 1] + output[k + two_dplus1 - 1];
    }
}

__global__ void downsweep_kernel(const int two_d, const int two_dplus1, int* input, int* output, const int N){
    unsigned int idx {blockIdx.x * blockDim.x + threadIdx.x};
    unsigned int k {two_dplus1 * idx};
    if (k < N) {
        int tmp {input[k + two_d - 1]};
        output[k + two_d - 1] = input[k + two_dplus1 - 1];
        output[k + two_dplus1 - 1] = tmp + input[k + two_dplus1 - 1];
    }
}

void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    
    // input and result are on device
    cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);

    // Up Sweep phase
    int rounded_length = nextPow2(N);
    for (int two_d = 1; two_d <= rounded_length / 2; two_d <<= 1) {
        int two_dplus1 = (two_d << 1);
        int threads_per_block {std::min(rounded_length / two_dplus1, THREADS_PER_BLOCK)};
        int num_blocks {(rounded_length / two_dplus1 + threads_per_block - 1) / threads_per_block};
        // dispatch to CUDA
        upsweep_kernel<<<num_blocks, threads_per_block>>>(two_d, two_dplus1, result, result, rounded_length);
        cudaDeviceSynchronize();
    }
    //Down Sweep pahse
    cudaMemset(&result[rounded_length - 1], 0, sizeof(int));
    for (int two_d = rounded_length / 2; two_d >= 1; two_d >>= 1) {
        int two_dplus1 = (two_d << 1);
        int threads_per_block {std::min(rounded_length / two_dplus1, THREADS_PER_BLOCK)};
        int num_blocks {(rounded_length / two_dplus1 + threads_per_block - 1) / threads_per_block};
        // dispatch to CUDA
        downsweep_kernel<<<num_blocks, threads_per_block>>>(two_d, two_dplus1, result, result, rounded_length);
        cudaDeviceSynchronize();
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;

    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found

__global__ void repetition_kernel(int* input, int* indicator, const int N) {
    unsigned int idx {blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < N - 1 && input[idx] == input[idx + 1]) {
        indicator[idx] = 1;
    } else {
        indicator[idx] = 0;
    }
}

__global__ void gather_kernel(int* exclusive_scan_results, int* output, const int N, int* total_count) {
    unsigned int idx {blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < N - 1) {
        if (exclusive_scan_results[idx] != exclusive_scan_results[idx + 1]) {
            output[exclusive_scan_results[idx]] = idx;
        }
    } else if (idx == N - 1) {
        *total_count = exclusive_scan_results[N - 1];
    }
}

int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    const int rounded_length {nextPow2(length)};

    // construct indicator array, whether input[i] == input[i + 1]
    int* device_indicator;
    cudaMalloc(&device_indicator, rounded_length * sizeof(int));
    int threads_per_block {std::min(rounded_length, THREADS_PER_BLOCK)};
    int num_blocks {(rounded_length + THREADS_PER_BLOCK - 1) / threads_per_block};
    repetition_kernel<<<num_blocks, threads_per_block>>>(device_input, device_indicator, length);

    // exclusive scan on indicator array
    // to get device_exclusive_scan_results
    int* device_exclusive_scan_results;
    cudaMalloc(&device_exclusive_scan_results, rounded_length * sizeof(int));
    exclusive_scan(device_indicator, length, device_exclusive_scan_results);
    cudaFree(device_indicator);

    // get repetition points in array    
    int* device_repetition_count;
    cudaMalloc(&device_repetition_count, sizeof(int));
    gather_kernel<<<num_blocks, threads_per_block>>>(device_exclusive_scan_results, device_output, length, device_repetition_count);
    cudaFree(device_exclusive_scan_results);

    // return results
    int output_length;
    cudaMemcpy(&output_length, device_repetition_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_repetition_count);
    
    return output_length; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
