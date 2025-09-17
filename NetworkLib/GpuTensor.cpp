#include "GpuTensor.h"

#include "kernel.h"

using namespace NetworkLib::Gpu;

void Environment::mse(const GpuView2& sought, const GpuView2& desired) {

	int size = sought.mView.extent(0);
	int batchSize = sought.mView.extent(1);
	Kernel::mse(mStream, sought.mGpu, desired.mGpu, mMseResult.mGpu, size, batchSize);
}
void Environment::relu(const GpuView1& o1, GpuView1& a1) {
	Kernel::relu(mStream, o1.mGpu, a1.mGpu, o1.mSize);
}
void Environment::applyReluPrime(const GpuView1& a1, GpuView1& p1) {
	Kernel::applyReluPrime(mStream, a1.mGpu, p1.mGpu, a1.mSize);
}
void Environment::softmax(const GpuView1& o1, GpuView1& a1) {
	Kernel::softmax(mStream, o1.mGpu, a1.mGpu, o1.mSize);
}
void Environment::batchedSoftmax(const GpuView2& o2, GpuView2& a2) {
	for( auto b : std::views::iota(0ULL, o2.mView.extent(1))) {
		auto o1 = o2.viewColumn(b);
		auto a1 = a2.viewColumn(b);
		softmax(o1, a1);
	}
}
void Environment::diff(const GpuView1& desired1, const GpuView1& sought1, GpuView1& primes1) {
	Kernel::diff(mStream, desired1.mGpu, sought1.mGpu, primes1.mGpu, desired1.mSize);
}
void Environment::updateWeights(const GpuView1& seen, GpuView2& weights, const GpuView1& primes, float learnRate) {
	
	int rows = weights.mView.extent(0);
	int cols = weights.mView.extent(1);

	Kernel::updateWeights(mStream, weights.mGpu, primes.mGpu, seen.mGpu, rows, cols, learnRate);

}
void Environment::batchedCopy(const GpuView2& source, GpuView2& dest) {
	auto size = source.mView.extent(0);
	auto batchSize = source.mView.extent(1);

	Kernel::batchedCopy(mStream, source.mGpu, dest.mGpu, size, batchSize);
}
void Environment::batchedBroadcast(const GpuView1& source, GpuView2& dest) {
	auto batchSize = dest.mView.extent(1);
	Kernel::batchedBroadcast(mStream, source.mGpu, dest.mGpu, source.mSize, batchSize);
}
void Environment::batchedBroadcastAdd(const GpuView1& source, GpuView2& dest) {
	auto batchSize = dest.mView.extent(1);
	Kernel::batchedBroadcastAdd(mStream, source.mGpu, dest.mGpu, source.mSize, batchSize);
}
void Environment::batchedDiff(const GpuView2& desired2, const GpuView2& sought2,GpuView2& primes2){
	Kernel::diff(mStream, desired2.mGpu, sought2.mGpu, primes2.mGpu, desired2.mSize);
}
void Environment::batchedUpdateWeights(const GpuView2& seen, GpuView2& weights, const GpuView2& primes, float learnRate) {

	int rows = weights.mView.extent(0);
	int cols = weights.mView.extent(1);
	int batchNum = seen.mView.extent(1);

	Kernel::batchedUpdateWeights(mStream, weights.mGpu, primes.mGpu, seen.mGpu, rows, cols, batchNum, learnRate);
}