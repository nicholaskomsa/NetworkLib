#include "Environment.h"

#include "kernel.h"

using namespace NetworkLib::Gpu;

void Environment::create() {
	Error::checkBlas(cublasCreate(&mHandle));
	Error::checkCuda(cudaStreamCreate(&mStream));
	Error::checkBlas(cublasSetStream(mHandle, mStream));

	mEnvironmentSpace.create(1);
	auto begin = mEnvironmentSpace.begin();
	mEnvironmentSpace.advance(mMseResult, begin);
}
void Environment::destroy() {

	mEnvironmentSpace.destroy();

	Error::checkCuda(cudaStreamDestroy(mStream));
	Error::checkBlas(cublasDestroy(mHandle));
}
cublasHandle_t Environment::getBlas() {
	return mHandle;
}
cudaStream_t Environment::getStream() {
	return mStream;
}
Environment::operator cudaStream_t() {
	return mStream;
}


void Environment::vecScale(GpuView1& a1, float scale) {
	auto result = cublasSscal(mHandle, a1.mSize, &scale, a1.mGpu, 1);
	Error::checkBlas(result);
}
void Environment::vecAddVec(const GpuView1& a1, GpuView1& o1) {
	float alpha = 1.0f;
	auto result = cublasSaxpy(mHandle, o1.mSize, &alpha, a1.mGpu, 1, o1.mGpu, 1);
	Error::checkBlas(result);
}
void Environment::matMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1) {
	//cuda is in C++ style of row-major
	//cublas wants Fortran style, col-major,
	//mdspan has been configured to be layout_left - cublas correct

	float alpha = 1.0f;
	float beta = 0.0f;

	Dimension r = w2.mView.extent(0)
		, c = w2.mView.extent(1);

	Error::checkMissMatch(c, i1.mSize);

	auto result = cublasSgemv(mHandle,
		CUBLAS_OP_N,
		r, c,
		&alpha,
		w2.mGpu, r,
		i1.mGpu, 1,
		&beta,
		o1.mGpu, 1);
	Error::checkBlas(result);
}
void Environment::batchedMatMulVec1(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {

	int c = w2.mView.extent(1);
	int k = i2.mView.extent(0);

	Error::checkMissMatch(c, k);

	Dimension batchSize = i2.mView.extent(1);

	for (auto b : std::views::iota(0ULL, batchSize)) {

		const GpuView1 i1 = i2.viewColumn(b);
		GpuView1 o1 = o2.viewColumn(b);

		matMulVec(w2, i1, o1);
	}
}
void Environment::batchedMatMulVec(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {

	float alpha = 1.0f;
	float beta = 0.0f;

	int r = w2.mView.extent(0);       // rows of matrix
	int c = w2.mView.extent(1);       // cols of matrix
	int batchSize = i2.mView.extent(1);

	Error::checkMissMatch(c, i2.mView.extent(0));
	Error::checkMissMatch(batchSize, o2.mView.extent(1));

	// Leading dimensions
	int lda = r;  // w2: (r × c)
	int ldb = c;  // i2: (c × 1)
	int ldc = r;  // o2: (r × 1)

	// Strides
	long long strideA = 0;                  // w2 is shared across batches
	long long strideB = static_cast<long long>(c);  // each vector is c × 1
	long long strideC = static_cast<long long>(r);  // each output is r × 1

	auto result = cublasSgemmStridedBatched(
		mHandle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		r, 1, c,
		&alpha,
		w2.mGpu, lda, strideA,
		i2.mGpu, ldb, strideB,
		&beta,
		o2.mGpu, ldc, strideC,
		batchSize
	);

	Error::checkBlas(result);
}
void Environment::batchedMatTMulVec(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {
	float alpha = 1.0f;
	float beta = 0.0f;

	int r = w2.mView.extent(0);       // rows of w2
	int c = w2.mView.extent(1);       // cols of w2
	int batchSize = i2.mView.extent(1);

	Error::checkMissMatch(r, i2.mView.extent(0));
	Error::checkMissMatch(batchSize, o2.mView.extent(1));

	// Leading dimensions
	int lda = r;  // w2: (r × c), column-major
	int ldb = r;  // i2: (r × 1), column-major
	int ldc = c;  // o2: (c × 1), column-major

	// Strides
	long long strideA = 0;                  // shared w2
	long long strideB = static_cast<long long>(r);  // input vector stride
	long long strideC = static_cast<long long>(c);  // output vector stride

	auto result = cublasSgemmStridedBatched(
		mHandle,
		CUBLAS_OP_T, CUBLAS_OP_N,  // transpose w2, no transpose i2
		c, 1, r,                   // output dim: (c × 1), inner dim: r
		&alpha,
		w2.mGpu, lda, strideA,
		i2.mGpu, ldb, strideB,
		&beta,
		o2.mGpu, ldc, strideC,
		batchSize
	);

	Error::checkBlas(result);
}

void Environment::matTMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1) {

	float alpha = 1.0f;
	float beta = 0.0f;

	Dimension r = w2.mView.extent(0)
		, c = w2.mView.extent(1);

	Error::checkMissMatch(r, i1.mSize);

	auto result = cublasSgemv(mHandle,
		CUBLAS_OP_T,
		r, c,
		&alpha,
		w2.mGpu, r,
		i1.mGpu, 1,
		&beta,
		o1.mGpu, 1);
	Error::checkBlas(result);
}

void Environment::mse(const GpuView2& sought, const GpuView2& desired) {

	int size = sought.mView.extent(0);
	int batchSize = sought.mView.extent(1);
	Kernel::mse(mStream, sought.mGpu, desired.mGpu, mMseResult.mGpu, size, batchSize);
}
float Environment::getMseResult() {
	mMseResult.downloadAsync(mStream);
	sync();
	return mMseResult;
}
void Environment::resetMseResult() {
	mMseResult = 0.0f;
	mMseResult.upload();
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
	for (auto b : std::views::iota(0ULL, o2.mView.extent(1))) {
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
void Environment::copy(const GpuView1& source, GpuView1& dest) {
	auto result = cublasScopy(mHandle, source.mSize, source.mGpu, 1, dest.mGpu, 1);
	Error::checkBlas(result);
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
void Environment::batchedDiff(const GpuView2& desired2, const GpuView2& sought2, GpuView2& primes2) {
	Kernel::diff(mStream, desired2.mGpu, sought2.mGpu, primes2.mGpu, desired2.mSize);
}
void Environment::batchedUpdateWeights(const GpuView2& seen, GpuView2& weights, const GpuView2& primes, float learnRate) {

	int rows = weights.mView.extent(0);
	int cols = weights.mView.extent(1);
	int batchNum = seen.mView.extent(1);

	Kernel::batchedUpdateWeights(mStream, weights.mGpu, primes.mGpu, seen.mGpu, rows, cols, batchNum, learnRate);
}

void Environment::activationFunction(LayerTemplate::ActivationFunction af, const GpuView1& o1, GpuView1& a1) {

	using ActivationFunction = LayerTemplate::ActivationFunction;
	switch (af) {
	case ActivationFunction::ReLU:
		relu(o1, a1);
		break;
	case ActivationFunction::Softmax:
		softmax(o1, a1);
		break;
	case ActivationFunction::None:
		copy(o1, a1);
		break;
	}
}
void Environment::batchedActivationFunction(LayerTemplate::ActivationFunction af, const GpuView2& o2, GpuView2& a2) {

	using ActivationFunction = LayerTemplate::ActivationFunction;
	switch (af) {
	case ActivationFunction::ReLU: {
		auto o1 = o2.flatten();
		auto a1 = a2.flatten();
		relu(o1, a1);
		break;
	}
	case ActivationFunction::Softmax:
		batchedSoftmax(o2, a2);
		break;
	case ActivationFunction::None:
		batchedCopy(o2, a2);
		break;
	}
}
void Environment::activationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView1& a1, GpuView1& p1) {

	using ActivationFunction = LayerTemplate::ActivationFunction;
	switch (af) {
	case ActivationFunction::ReLU:
		applyReluPrime(a1, p1);
		break;
	case ActivationFunction::None:
		break;
	}
}
void Environment::batchedActivationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView2& a2, GpuView2& p2) {

	using ActivationFunction = LayerTemplate::ActivationFunction;
	switch (af) {
	case ActivationFunction::ReLU: {
		auto p1 = p2.flatten();
		auto a1 = a2.flatten();
		applyReluPrime(a1, p1);
		break;
	}
	case ActivationFunction::None:
		break;
	}
}

void Environment::errorFunction(LayerTemplate::ActivationFunction af, const GpuView1& desired, const GpuView1& sought, GpuView1& p1) {
	switch (af) {
	case LayerTemplate::ActivationFunction::Softmax:
		//softmax-cross-entropy is a diff
		[[fallthrouh]];
	case LayerTemplate::ActivationFunction::None:
		[[fallthrouh]];
	default:
		diff(desired, sought, p1);
		return;
	}
}
void Environment::batchedErrorFunction(LayerTemplate::ActivationFunction af, const GpuView2& desired2, const GpuView2& sought2, GpuView2& p2) {
	switch (af) {
	case LayerTemplate::ActivationFunction::Softmax:
		//softmax-cross-entropy is a diff
		[[fallthrouh]];
	case LayerTemplate::ActivationFunction::None:
		[[fallthrouh]];
	default:
		batchedDiff(desired2, sought2, p2);
		return;
	}
}
void Environment::sync() {
	Error::checkCuda(cudaDeviceSynchronize());
}


void Environment::example() {

	Environment gpu;
	gpu.create();

	Gpu::FloatSpace1 fs1;
	Gpu::GpuView1 b1;
	Gpu::GpuView2 o2, a2, i2, d2, p2;
	Gpu::GpuView2 w2;
	Gpu::GpuView2 activations, softmax;

	std::size_t inputSize = 3
		, biasSize = 2
		, batchSize = 1;

	fs1.create((inputSize + biasSize * 4) * batchSize
		+ biasSize
		+ biasSize * inputSize + biasSize * 2 * batchSize);

	auto begin = fs1.begin();

	fs1.advance(i2, begin, inputSize, batchSize);
	fs1.advance(w2, begin, biasSize, inputSize);
	fs1.advance(b1, begin, biasSize);
	fs1.advance(o2, begin, biasSize, batchSize);
	fs1.advance(a2, begin, biasSize, batchSize);
	fs1.advance(p2, begin, biasSize, batchSize);
	fs1.advance(d2, begin, biasSize, batchSize);
	fs1.advance(softmax, begin, biasSize, batchSize);
	fs1.advance(activations, begin, biasSize, batchSize);

	std::fill(w2.begin(), w2.end(), 1);
	std::fill(i2.begin(), i2.end(), 1);
	std::fill(b1.begin(), b1.end(), 1);
	std::fill(d2.begin(), d2.end(), 0.5);

	i2.mView[0, 1] = 0;

	d2.mView[0, 0] = .314;
	d2.mView[0, 1] = 1;
	d2.mView[0, 2] = 0;

	activations.mView[0, 0] = 0.3;
	activations.mView[1, 0] = -0.8;

	activations.mView[0, 1] = 0.9;
	activations.mView[1, 1] = 0.1;

	fs1.upload();

	gpu.sync();

	for (auto generation : std::views::iota(0, 5000)) {

		auto af = LayerTemplate::ActivationFunction::Softmax;

		auto forward = [&]() {

			gpu.batchedMatMulVec(w2, i2, o2);
			gpu.batchedBroadcastAdd(b1, o2);
			gpu.batchedActivationFunction(af, o2, a2);

			};
		forward();

		auto backward = [&]() {

			gpu.batchedErrorFunction(af, d2, a2, p2);

			gpu.batchedUpdateWeights(i2, w2, p2, 0.002);
			};
		backward();
	}
	gpu.batchedActivationFunction(LayerTemplate::ActivationFunction::Softmax, activations, softmax);

	fs1.downloadAsync(gpu);

	gpu.sync();

	for (const auto& f : softmax)
		std::print("{} ", f);
	std::println("");
	fs1.destroy();

	gpu.destroy();
}
