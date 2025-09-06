#pragma once

namespace NetworkLib {
	namespace Gpu {
		namespace Kernel{
 
			void relu(cudaStream_t stream, const float* outputs, float* reluActivations, int size);
            void applyReluPrime(cudaStream_t stream, const float* reluActivations, float* primes, int size);
			void updateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int cols, int rows, float learnRate);

			void softmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size);
           
			
			void diff(cudaStream_t stream, const float* desired, const float* sought, float* primes, int size);
		}
	}
}

