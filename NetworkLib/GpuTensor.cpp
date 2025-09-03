#include "GpuTensor.h"

#include "kernel.h"

using namespace NetworkLib::Gpu;

void Environment::relu(const GpuView1& o1, GpuView1& a1) {
	Kernel::relu(getStream(), o1.mGpu, a1.mGpu, o1.mSize);
}