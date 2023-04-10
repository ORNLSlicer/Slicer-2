//! \author Charles Wade
/*!
CUDA IMPLEMENTATION OF A TRAVELING SALESMAN SOLVER

This algorithm processes the exact solution to Traveling Salesman Problem using a CUDA GPU. It is comprized of two major steps: 
1. Generate Permutations/ Calculate Distances.
2. Perform a global reduction to find global min or max.

\note See design doc in repo for more information
*/

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

//! \macro Used to catch CUDA errors. Must be used on all non-kernel launch CUDA calls.
#define CUDA_SAFE(x_) {cudaError_t cudaStatus = x_; if (cudaStatus != cudaSuccess) {fprintf(stderr, "Error  %d - %s\n", cudaStatus, cudaGetErrorString(cudaStatus));}}

//! \macro Used to catch memory errors. Should be use on all allocation calls.
#define SAFE(x_) {if((x_) == NULL) printf("out of memory. %d\n", __LINE__);}

#include "managers/gpu_manager.h"

namespace ORNL
{
    //! \brief A host function to compute N!
    __host__
    unsigned long long h_factorial(uint8_t n)
    {

        unsigned long long factorial = 1;

        for (int i = 1; i <= n; i++)
            factorial = factorial * i;

        return factorial;
    }

    //! \brief A function to convert N from the decimal to factorial number system
    __host__
    void h_compute_factoradic(unsigned long long n, uint8_t size, uint8_t * factoradic)
    {
        uint8_t count = 1;
        while (n != 0) {
            factoradic[size - count] = n % count;
            n = n / count;
            count++;
        }
    }

    //! \brief A function to compute the Nth permutation directly
    __host__
    int * h_compute_nth_permutation(unsigned long long n, int start, int width, int * permutation)
    {
        uint8_t * factoradic;
        SAFE(factoradic = static_cast<uint8_t *>(calloc(width - 1, sizeof(uint8_t))));

        h_compute_factoradic(n, width-1, factoradic);

        //! \note Compute the reference row, omitting the start island
        uint8_t * referenceRow;
        SAFE(referenceRow =  static_cast<uint8_t *>(malloc((width - 1) * sizeof(uint8_t))));
        for(uint8_t i = 0, j = 0; i < width;i++)
        {
            if(i != start)
            {
                referenceRow[j] = i;
                j++;
            }
        }

        //! \note Sets our chosen start point
        permutation[0] = start;

        for(uint8_t i = 1; i < width; i++)
        {
            uint8_t index = factoradic[i-1];
            permutation[i] = (int)referenceRow[index];
            for (uint8_t j = index; j < width-2; j++)
            {
                referenceRow[j] = referenceRow[j + 1];
            }

        }

        return permutation;
    }

    //! \brief A function to compute the Nth permutation directly
    __device__
    void d_compute_factoradic(unsigned long long n, uint8_t size, uint8_t * factoradic)
    {
        uint8_t count = 1;
        while (n != 0)
        {
            factoradic[size - count] = n % count;
            n = n / count;
            count++;
        }
    }

    //! \brief A kernel to calculate a permutation's distance
    __global__
    void calculate_distances(uint8_t width, unsigned long long height, uint8_t start, float * x, float * y, float * distances, unsigned long long permutations_completed)
    {
        unsigned long long threadId = blockIdx.x*blockDim.x+threadIdx.x;

        //! \note Only compute compute if the threadID is within our target range.
        if(threadId < height)
        {
            //! \note CUDA does not implement calloc so we use malloc and then manually clear the entries
            uint8_t * factoradic;
            SAFE(factoradic = static_cast<uint8_t *>(malloc((width - 1) * sizeof(uint8_t))));            
            for(int i = 0;i < width-1;i++)
            {
                factoradic[i] = 0;
            }

            //! \note Compute and store our reference row.
            uint8_t * referenceRow;
            SAFE(referenceRow = static_cast<uint8_t *>(malloc((width - 1) * sizeof(uint8_t))));
            for(uint8_t i = 0, j = 0; i < width;i++)
            {
                if(i != start)
                {
                    referenceRow[j] = i;
                    j++;
                }
            }

            d_compute_factoradic(threadId + permutations_completed, (width - 1), factoradic);

            /*! \note We get to pick where we start and don't actually need to return to it after visiting everything else. 
                      This reduces the number of permutations from N! to (N-1)!. */
            uint8_t lastIndex = start;
            float totalDistance = 0.0;

            for(uint8_t i = 1; i < width; i++)
            {
                uint8_t fractionalFactoradic = factoradic[i-1];
                uint8_t nextIndex = referenceRow[fractionalFactoradic];

                //! \note Calculate 2D distance and add to total
                totalDistance += sqrtf(powf(x[nextIndex] - x[lastIndex],2) + powf(y[nextIndex] - y[lastIndex],2));
                lastIndex = nextIndex;
                
                //! \note Move element in row left at an index if needed.
                for (uint8_t j = fractionalFactoradic; j < (width - 2); j++)
                {
                    referenceRow[j] = referenceRow[j + 1];
                }
            }

            distances[threadId] = totalDistance;

            free(referenceRow);
            free(factoradic);
        }
    }
    //! \brief The driver function
    int * compute_tsp(float * h_x, float * h_y, int size, int startIndex, bool shortest)
    {
        /*------Set Up CUDA Device-----*/
        if(GPU->count() > 0)
            cudaSetDevice(0);

        size_t freeBytes;
        size_t totalBytes;

        /*------Host Variables-----*/
        uint8_t n = static_cast<uint8_t>(size);
        uint8_t width = n;
        unsigned long long totalHeight = h_factorial(n - 1);
        unsigned long long permutationsCompleted = 0;

        //! \note The memory our x/y points take on the GPU stored as floats
        size_t bytesOfPoints = width * sizeof(float) * 2;

        thrust::host_vector<float> h_computedDistances;
        thrust::host_vector<unsigned long long> h_computedPermutations;

        /*------Compute Loop-----*/
        //! \note Used to break-up the computation into sizes that will fit on the GPU's memory
        while(permutationsCompleted < totalHeight)
        {
            unsigned long long height;

            size_t bytesOfDistancesLeft = (totalHeight - permutationsCompleted) * sizeof(float);
            size_t totalBytesLeftToAllocate = bytesOfDistancesLeft + bytesOfPoints;
            
            //! \note Fetches free and total memory to calculate allocations
            cudaMemGetInfo(&freeBytes, &totalBytes);
            if(totalBytesLeftToAllocate > freeBytes)
            {
                // Calculates how many permutations can fit within the free memory
                height = ((freeBytes - bytesOfPoints) / sizeof(float));
            }else
            {
                // Sets to how ever many permutations are left to do
                height = totalHeight - permutationsCompleted;
            }

            /*------Host Variables-----*/
            size_t widthBytes = width * sizeof(float);
            size_t heightBytes = height * sizeof(float);

            //Block/Grid Sizes
            int blockSize, gridSize;
            blockSize = GPU->getDeviceInformation(0).maxThreadsPerBlock;
            gridSize = static_cast<int>(ceil((float)height / blockSize));

             /*------Device Variables-----*/
            float * d_x;
            float * d_y;
            float * d_distances;

            /*------Device Memory Allocation-----*/
            CUDA_SAFE(cudaMalloc(&d_x, widthBytes));
            CUDA_SAFE(cudaMalloc(&d_y, widthBytes));
            CUDA_SAFE(cudaMalloc(&d_distances, heightBytes));

            /*------Host to Device Data Copy-----*/
            CUDA_SAFE(cudaMemcpy(d_x, h_x, widthBytes, cudaMemcpyHostToDevice));
            CUDA_SAFE(cudaMemcpy(d_y, h_y, widthBytes, cudaMemcpyHostToDevice));

            /*------Calculate Distances-----*/
            calculate_distances<<<gridSize, blockSize>>>(width, height, startIndex, d_x, d_y, d_distances, permutationsCompleted);
            CUDA_SAFE(cudaGetLastError());
            //! \note Wait for all threads to finish
            CUDA_SAFE(cudaDeviceSynchronize());

            //! \note Free points from memory
            CUDA_SAFE(cudaFree(d_x));
            CUDA_SAFE(cudaFree(d_y));

            /*------Reduction-----*/
            thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_distances);
            thrust::device_ptr<float> target_ptr = thrust::min_element(dev_ptr, dev_ptr + height);
            h_computedDistances.push_back(target_ptr[0]);
            h_computedPermutations.push_back(&target_ptr[0] - &dev_ptr[0]);
            
            CUDA_SAFE(cudaFree(d_distances));
            //! \note Add how ever many permutations we just computed to our total
            permutationsCompleted += height;
        }
        /*------Perform Final Parallel Reduction-----*/
        thrust::device_vector<float> d_computedDistances = h_computedDistances;
        thrust::device_vector<float>::iterator iter;
        
        if(shortest)
            iter = thrust::min_element(d_computedDistances.begin(), d_computedDistances.end());
        else
            iter = thrust::max_element(d_computedDistances.begin(), d_computedDistances.end());
        
        unsigned long long targetIndex = iter - d_computedDistances.begin();

        unsigned long long targetPermutationID = h_computedPermutations[targetIndex];
        
        //! \note Calculate the Nth permutation on the Host
        int * optimizedPath = static_cast<int*>(malloc(width * sizeof(int)));
        h_compute_nth_permutation(targetPermutationID, startIndex, width, optimizedPath);

        return optimizedPath;
    }
}

