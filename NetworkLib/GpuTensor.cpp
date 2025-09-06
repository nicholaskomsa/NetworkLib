#include "GpuTensor.h"

#include "kernel.h"

using namespace NetworkLib::Gpu;

void Environment::relu(const GpuView1& o1, GpuView1& a1) {
	Kernel::relu(getStream(), o1.mGpu, a1.mGpu, o1.mSize);
}
void Environment::applyReluPrime(const GpuView1& a1, GpuView1& p1) {
	Kernel::applyReluPrime(getStream(), a1.mGpu, p1.mGpu, a1.mSize);
}
void Environment::softmax(const GpuView1& o1, GpuView1& a1) {
	Kernel::softmax(getStream(), o1.mGpu, a1.mGpu, o1.mSize);
}
void Environment::diff(const GpuView1& desired1, const GpuView1& sought1, GpuView1& primes1) {
	Kernel::diff(getStream(), desired1.mGpu, sought1.mGpu, primes1.mGpu, desired1.mSize);
}
void Environment::updateWeights(Environment& env, const GpuView1& seen, GpuView2& weights, const GpuView1& primes, float learnRate) {
	
	int rows = weights.mView.extent(0);
	int cols = weights.mView.extent(1);

	Kernel::updateWeights(getStream(), weights.mGpu, primes.mGpu, seen.mGpu, rows, cols, learnRate);

}