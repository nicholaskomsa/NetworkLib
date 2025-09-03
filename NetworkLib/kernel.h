#pragma once

namespace NetworkLib {
	namespace Gpu {
		namespace Kernel{
			void relu(cudaStream_t stream, float* outputs, float* activations, int size);
		}
	}
}

