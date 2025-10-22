#include "Environment.h"

#include "kernel.h"

#include <numeric>
#include <random>
#include "CpuNetwork.h"

using namespace NetworkLib::Gpu;

std::atomic<std::size_t> Environment::mCommandCounter;
std::mutex Environment::mCommandMutex;

void Environment::create() {
	Error::checkBlas(cublasCreate(&mHandle));
	Error::checkCuda(cudaStreamCreate(&mStream));
	Error::checkBlas(cublasSetStream(mHandle, mStream));

	commandQueueSync(3);

	mLinkedFloatSpace.create(2);

	auto& gpuSpace = mLinkedFloatSpace.mGpuSpace;

	auto begin = gpuSpace.begin();
	gpuSpace.advance(mMseResult, begin);
	gpuSpace.advance(mMissesResult, begin);

}
void Environment::destroy() {

	mLinkedFloatSpace.destroy();

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


void Environment::conv1(const GpuView3& w3, const GpuView1& i1, GpuView1& o1) {

	auto kernelSize = w3.mView.extent(0);
	auto kernelDepth = w3.mView.extent(2);

	Kernel::conv1(mStream, w3.mGpu, o1.mGpu, i1.mGpu, i1.mView.extent(0), kernelSize, kernelDepth);

	commandQueueSync();
}
void Environment::batchedConv1(const GpuView3& w3, const GpuView2& i2, GpuView2& o2) {

	std::size_t batchSize = i2.mView.extent(1);
	for (auto b : std::views::iota(0ULL, batchSize)) {
		GpuView1 i1 = i2.viewColumn(b);
		GpuView1 o1 = o2.viewColumn(b);
		conv1(w3, i1, o1);
	}
}
void Environment::conv1UpdateKernel( GpuView3& w3, const GpuView1& i1, const GpuView1& p1, float learnRate) {

	std::size_t kernelNum = w3.mView.extent(2);
	std::size_t kernelSize = w3.mView.extent(0);
	Kernel::conv1UpdateKernel(mStream, w3.mGpu, p1.mGpu, i1.mGpu, i1.mView.extent(0), kernelSize, kernelNum, learnRate);
	commandQueueSync();
}
void Environment::batchedConv1UpdateKernel( GpuView3& w3, const GpuView2& i2, const GpuView2& p2, float learnRate) {
	std::size_t batchSize = i2.mView.extent(1);
	for (auto b : std::views::iota(0ULL, batchSize)) {
		GpuView1 i1 = i2.viewColumn(b);
		GpuView1 p1 = p2.viewColumn(b);
		conv1UpdateKernel(w3, i1, p1, learnRate);
	}
}
void Environment::vecScale(GpuView1& a1, float scale) {

	auto result = cublasSscal(mHandle, a1.mSize, &scale, a1.mGpu, 1);
	Error::checkBlas(result);
	commandQueueSync();
}
void Environment::vecAddVec(const GpuView1& a1, GpuView1& o1) {
	float alpha = 1.0f;
	auto result = cublasSaxpy(mHandle, o1.mSize, &alpha, a1.mGpu, 1, o1.mGpu, 1);
	Error::checkBlas(result);
	commandQueueSync();
}
void Environment::matMulVec(const GpuView3& w3, const GpuView1& i1, GpuView1& o1) {
	//cuda is in C++ style of row-major
	//cublas wants Fortran style, col-major,
	//mdspan has been configured to be layout_left - cublas correct

	float alpha = 1.0f;
	float beta = 0.0f;

	Dimension r = w3.mView.extent(0)
		, c = w3.mView.extent(1);

	Error::checkMissMatch(c, i1.mSize);

	auto result = cublasSgemv(mHandle,
		CUBLAS_OP_N,
		r, c,
		&alpha,
		w3.mGpu, r,
		i1.mGpu, 1,
		&beta,
		o1.mGpu, 1);
	Error::checkBlas(result);
	commandQueueSync();
}
void Environment::batchedMatMulVec1(const GpuView3& w3, const GpuView2& i2, GpuView2& o2) {

	int c = w3.mView.extent(1);
	int k = i2.mView.extent(0);

	Error::checkMissMatch(c, k);

	Dimension batchSize = i2.mView.extent(1);

	for (auto b : std::views::iota(0ULL, batchSize)) {

		const GpuView1 i1 = i2.viewColumn(b);
		GpuView1 o1 = o2.viewColumn(b);

		matMulVec(w3, i1, o1);
	}
	commandQueueSync();
}
void Environment::batchedMatMulVec(const GpuView3& w3, const GpuView2& i2, GpuView2& o2) {

	float alpha = 1.0f, beta = 0.0f;

	int r = w3.mView.extent(0)       // rows of matrix
		, c = w3.mView.extent(1)       // cols of matrix
		, batchSize = i2.mView.extent(1);

	Error::checkMissMatch(c, i2.mView.extent(0));
	Error::checkMissMatch(batchSize, o2.mView.extent(1));

	// Leading dimensions
	int lda = r  // w2: (r × c)
		, ldb = c  // i2: (c × 1)
		, ldc = r;  // o2: (r × 1)

	// Strides
	long long strideA = 0                  // w2 is shared across batches
		, strideB = static_cast<long long>(c)  // each vector is c × 1
		, strideC = static_cast<long long>(r);  // each output is r × 1

	auto result = cublasSgemmStridedBatched(
		mHandle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		r, 1, c,
		&alpha,
		w3.mGpu, lda, strideA,
		i2.mGpu, ldb, strideB,
		&beta,
		o2.mGpu, ldc, strideC,
		batchSize
	);

	Error::checkBlas(result);
	commandQueueSync();
}
void Environment::batchedMatTMulVec(const GpuView3& w3, const GpuView2& i2, GpuView2& o2) {
	float alpha = 1.0f;
	float beta = 0.0f;

	int r = w3.mView.extent(0);       // rows of w2
	int c = w3.mView.extent(1);       // cols of w2
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
		w3.mGpu, lda, strideA,
		i2.mGpu, ldb, strideB,
		&beta,
		o2.mGpu, ldc, strideC,
		batchSize
	);

	Error::checkBlas(result);
	commandQueueSync();
}

void Environment::backwardConv1(const GpuView3& w3, const GpuView1& e1, GpuView1& p1) {
	
	auto wView = w3.mView;
	auto eView = e1.mView;
	auto pView = p1.mView;

	auto kernelDepth = wView.extent(2);
	auto errorsPerKernel = eView.extent(0) / kernelDepth;
	auto kernelSize = wView.extent(0);

	for( auto k : std::views::iota(0ULL, kernelDepth))
		for (auto n : std::views::iota(0ULL, errorsPerKernel)){
			auto sum = 0.0f;
			auto idx = n + k * errorsPerKernel;
			auto e = eView[idx];

			for( auto w : std::views::iota(0ULL, kernelSize))
				sum += wView[w,0,k] * e;
		
			pView[idx] = sum;
		}
}
void Environment::batchedBackwardConv1(const GpuView3& w3, const GpuView2& o2, GpuView2& p2) {
	
	w3.downloadAsync(mStream); sync();
	o2.downloadAsync(mStream); sync();

	Dimension batchSize = o2.mView.extent(1);

	for (auto b : std::views::iota(0ULL, batchSize)) {

		GpuView1 o1 = o2.viewColumn(b);
		GpuView1 p1 = p2.viewColumn(b);

		backwardConv1(w3, o1, p1);
	}

	p2.upload();
}
void Environment::matTMulVec(const GpuView3& w3, const GpuView1& i1, GpuView1& o1) {

	float alpha = 1.0f;
	float beta = 0.0f;

	Dimension r = w3.mView.extent(0)
		, c = w3.mView.extent(1);

	Error::checkMissMatch(r, i1.mSize);

	auto result = cublasSgemv(mHandle,
		CUBLAS_OP_T,
		r, c,
		&alpha,
		w3.mGpu, r,
		i1.mGpu, 1,
		&beta,
		o1.mGpu, 1);
	Error::checkBlas(result);
	commandQueueSync();
}

void Environment::score(const GpuView2& sought, const GpuView2& desired) {
	int size = sought.mView.extent(0);
	int batchSize = sought.mView.extent(1);
	Kernel::score(mStream, sought.mGpu, desired.mGpu, mMissesResult.mGpu, size, batchSize);
	commandQueueSync();
}
void Environment::mse(const GpuView2& sought, const GpuView2& desired) {

	int size = sought.mView.extent(0);
	int batchSize = sought.mView.extent(1);
	Kernel::mse(mStream, sought.mGpu, desired.mGpu, mMseResult.mGpu, size, batchSize);
	commandQueueSync();
}
float Environment::getMseResult() {
	return mMseResult;
}
void Environment::resetMseResult() {
	mMseResult = 0.0f;
	mMseResult.upload();
}
void Environment::resetMissesResult() {
	mMissesResult = 0;;
	mMissesResult.upload();
}
int Environment::getMissesResult() {
	return mMissesResult;
}

void Environment::downloadConvergenceResults(bool doSync) {
	mMissesResult.downloadAsync(mStream);
	mMseResult.downloadAsync(mStream);
	if (doSync)
		sync();
	commandQueueSync(2);
}
void Environment::relu(const GpuView1& o1, GpuView1& a1) {
	Kernel::relu(mStream, o1.mGpu, a1.mGpu, o1.mSize);
	commandQueueSync();
}
void Environment::applyReluPrime(const GpuView1& a1, GpuView1& p1) {
	Kernel::applyReluPrime(mStream, a1.mGpu, p1.mGpu, a1.mSize);
	commandQueueSync();
}
void Environment::softmax(const GpuView1& o1, GpuView1& a1) {
	Kernel::softmax(mStream, o1.mGpu, a1.mGpu, o1.mSize);
	commandQueueSync();
}
void Environment::batchedSoftmax1(const GpuView2& o2, GpuView2& a2) {
	for (auto b : std::views::iota(0ULL, o2.mView.extent(1))) {
		auto o1 = o2.viewColumn(b);
		auto a1 = a2.viewColumn(b);
		softmax(o1, a1);
	}
}
void Environment::batchedSoftmax(const GpuView2& o2, GpuView2& a2) {

	auto size = o2.mView.extent(0);
	auto batchSize = o2.mView.extent(1);

	Kernel::batchedSoftmax(mStream, o2.mGpu, a2.mGpu, size, batchSize );
	commandQueueSync();
}
void Environment::diff(const GpuView1& desired1, const GpuView1& sought1, GpuView1& primes1) {
	Kernel::diff(mStream, desired1.mGpu, sought1.mGpu, primes1.mGpu, desired1.mSize);
	commandQueueSync();
}
void Environment::updateWeights(const GpuView1& seen, GpuView3& weights, const GpuView1& primes, float learnRate) {

	int rows = weights.mView.extent(0);
	int cols = weights.mView.extent(1);

	Kernel::updateWeights(mStream, weights.mGpu, primes.mGpu, seen.mGpu, rows, cols, learnRate);
	commandQueueSync();
}
void Environment::copy(const GpuView1& source, GpuView1& dest) {
	auto result = cublasScopy(mHandle, source.mSize, source.mGpu, 1, dest.mGpu, 1);
	Error::checkBlas(result);
	commandQueueSync();
}
void Environment::batchedCopy(const GpuView2& source, GpuView2& dest) {
	auto size = source.mView.extent(0);
	auto batchSize = source.mView.extent(1);

	Kernel::batchedCopy(mStream, source.mGpu, dest.mGpu, size, batchSize);
	commandQueueSync();
}
void Environment::batchedBroadcast(const GpuView1& source, GpuView2& dest) {
	auto batchSize = dest.mView.extent(1);
	Kernel::batchedBroadcast(mStream, source.mGpu, dest.mGpu, source.mSize, batchSize);
	commandQueueSync();
}
void Environment::batchedBroadcastAdd(const GpuView1& source, GpuView2& dest) {
	auto batchSize = dest.mView.extent(1);
	Kernel::batchedBroadcastAdd(mStream, source.mGpu, dest.mGpu, source.mSize, batchSize);
	commandQueueSync();
}
void Environment::batchedDiff(const GpuView2& desired2, const GpuView2& sought2, GpuView2& primes2) {
	Kernel::diff(mStream, desired2.mGpu, sought2.mGpu, primes2.mGpu, desired2.mSize);
	commandQueueSync();
}
void Environment::batchedUpdateWeights(const GpuView2& seen, GpuView3& weights, const GpuView2& primes, float learnRate) {

	int rows = weights.mView.extent(0);
	int cols = weights.mView.extent(1);
	int batchNum = seen.mView.extent(1);

	Kernel::batchedUpdateWeights(mStream, weights.mGpu, primes.mGpu, seen.mGpu, rows, cols, batchNum, learnRate);
	commandQueueSync();
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
void Environment::activationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView1& o1, GpuView1& p1) {

	using ActivationFunction = LayerTemplate::ActivationFunction;
	switch (af) {
	case ActivationFunction::ReLU:
		applyReluPrime(o1, p1);
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
void Environment::deviceSync() {
	Error::checkCuda(cudaDeviceSynchronize());
	mCommandCounter = 0;
}
void Environment::sync() {
	Error::checkCuda(cudaStreamSynchronize(mStream));
}
void Environment::commandQueueSync(std::size_t commandCount) {

	mCommandCounter += commandCount;
	
	if (mCommandCounter < mMaxQueuedCommands) return;

	std::scoped_lock lock(mCommandMutex);

	if (mCommandCounter >= mMaxQueuedCommands) 
		deviceSync();
}

void Environment::example() {

	Environment gpu;
	gpu.create();

	LinkedFloatSpace linkedSpace;
	auto& [cpuSpace, gpuSpace] = linkedSpace;

	Gpu::GpuView1 b1;
	Gpu::GpuView2 o2, a2, i2, d2, p2;
	Gpu::GpuView3 w3;
	Gpu::GpuView2 activations, softmax;

	std::size_t inputSize = 3
		, biasSize = 2
		, batchSize = 3;

	linkedSpace.create((inputSize + biasSize * 4) * batchSize
		+ biasSize
		+ biasSize * inputSize + biasSize * 2 * batchSize);

	auto begin = gpuSpace.begin();
	
	gpuSpace.advance(i2, begin, inputSize, batchSize);
	gpuSpace.advance(w3, begin, biasSize, inputSize, 1ULL);
	gpuSpace.advance(b1, begin, biasSize);
	gpuSpace.advance(o2, begin, biasSize, batchSize);
	gpuSpace.advance(a2, begin, biasSize, batchSize);
	gpuSpace.advance(p2, begin, biasSize, batchSize);
	gpuSpace.advance(d2, begin, biasSize, batchSize);
	gpuSpace.advance(softmax, begin, biasSize, batchSize);
	gpuSpace.advance(activations, begin, biasSize, batchSize);
	
	std::fill(w3.begin(), w3.end(), 1);
	std::fill(i2.begin(), i2.end(), 1);
	std::fill(b1.begin(), b1.end(), 1);
	std::fill(d2.begin(), d2.end(), 0.5);

	i2.mView[0ULL, 1ULL] = 0;

	d2.mView[0, 0] = .314;
	d2.mView[0, 1] = 1;
	d2.mView[0, 2] = 0;

	activations.mView[0, 0] = 0.3;
	activations.mView[1, 0] = -0.8;

	activations.mView[0, 1] = 0.9;
	activations.mView[1, 1] = 0.1;

	gpuSpace.upload();

	gpu.sync();

	for (auto generation : std::views::iota(0, 5000)) {

		auto af = LayerTemplate::ActivationFunction::Softmax;

		auto forward = [&]() {

			gpu.batchedMatMulVec(w3, i2, o2);
			gpu.batchedBroadcastAdd(b1, o2);
			gpu.batchedActivationFunction(af, o2, a2);

			};
		forward();

		auto backward = [&]() {

			gpu.batchedErrorFunction(af, d2, a2, p2);

			gpu.batchedUpdateWeights(i2, w3, p2, 0.002);
			};
		backward();
	}
	gpu.batchedActivationFunction(LayerTemplate::ActivationFunction::Softmax, activations, softmax);

	gpuSpace.downloadAsync(gpu);

	gpu.sync();

	for (const auto& f : softmax)
		std::print("{} ", f);
	std::println("");
	
	linkedSpace.destroy();

	gpu.destroy();
}
void Environment::example2() {

	Environment gpu;
	gpu.create();
	 
	LinkedFloatSpace linkedSpace;
	auto& [cpuSpace, gpuSpace] = linkedSpace;

	using GpuSample = std::pair< GpuView1, GpuView1>;
	std::vector<GpuSample> samples;
;

	Gpu::GpuView1 b1, p1, o1, a1, e1;
	Gpu::GpuView3 w3;

	std::size_t kernelSize = 7;
	std::size_t inputSize = 10
		, biasSize = inputSize - kernelSize + 1
		, outputSize = biasSize;

	//1 sample per input
	samples.resize(inputSize);

	linkedSpace.create((inputSize + outputSize)*samples.size() + biasSize * 4 + kernelSize);

	auto begin = gpuSpace.begin();
	
	for (auto& [seen,desired] : samples) {
		gpuSpace.advance(seen, begin, inputSize);
		gpuSpace.advance(desired, begin, outputSize);
	}

	gpuSpace.advance(w3, begin, kernelSize, 1ULL, 1ULL);
	gpuSpace.advance(b1, begin, biasSize);
	gpuSpace.advance(p1, begin, biasSize);
	gpuSpace.advance(a1, begin, biasSize);
	gpuSpace.advance(e1, begin, biasSize);
	gpuSpace.advance(o1, begin, outputSize);

	std::uniform_real_distribution reals(-1.0f, 1.0f), noise(0.0f, 0.1f);
	std::mt19937 random;
	std::size_t sampleIdx = 0;

 	auto createPingSample = [&](auto& seen, auto& desired) {

 		std::size_t idx = sampleIdx++;
		
		std::fill(desired.begin(), desired.end(), 0);

		//static
		std::generate(seen.begin(), seen.end(), [&]() { return 0; });// noise(random); });

		//ping
		seen.mView[idx] = 1.0f;
		
		std::size_t minIdx = std::max<int>(0, idx - kernelSize+1);
		std::size_t maxIdx = std::min(outputSize, idx + 1);

		for (auto i : std::views::iota(minIdx, maxIdx )) 
			desired.mView[i] = 1.0f;
		
		};

	for (auto& [seen, desired] : samples) 
		createPingSample(seen, desired);
	

	auto createKernel = [&]() {
		std::generate(w3.begin(), w3.end(), [&]() {return reals(random); });
		std::generate(b1.begin(), b1.end(), [&]() {return reals(random); });
		};
	createKernel();



	gpuSpace.upload();
	
	gpu.sync();

	auto print = [&](std::string_view caption, const Gpu::GpuView1& v1) {
		v1.downloadAsync(gpu); gpu.sync();
		std::print("{}: ", caption);
		for (const auto& val : v1)
			std::print("{} ", val);
		std::println("\n");
		};


	auto forward = [&](auto& i1) {

		//forward
		gpu.conv1(w3, i1, o1);
		gpu.vecAddVec(b1, o1);
		gpu.activationFunction(LayerTemplate::ActivationFunction::ReLU, o1, a1);

		};

	auto convergence = [&](bool print=false) {

		float mse = 0.0f;
		for( const auto& [seen, desired] : samples){
			forward(seen);
			a1.downloadAsync(gpu); gpu.sync();

			auto& sought = a1;
			mse += Cpu::Network::mse(sought.mView, desired.mView);
		
			if (print) 
				std::println("\nseen: {}\ndesired: {}\nsought: {}\n", seen, desired, sought);
		}

		std::println("convergence: {}", mse);
		};
	auto backward = [&](auto& i1, auto& d1) {

		gpu.errorFunction(LayerTemplate::ActivationFunction::None, d1, a1, e1);

		e1.downloadAsync(gpu); gpu.sync();
		gpu.backwardConv1(w3, e1, p1);
		p1.upload(); gpu.sync();

		gpu.activationFunctionPrime(LayerTemplate::ActivationFunction::ReLU, o1, p1);
		gpu.conv1UpdateKernel(w3, i1, p1, 0.002f);
		};

	for( auto generation : std::views::iota(0, 1000)) {
		
		auto& [seen, desired] = samples[generation % samples.size()];

		forward(seen);
		backward(seen, desired);
		convergence();
	}

	convergence(true);

	gpuSpace.downloadAsync(gpu);
	gpu.sync();
	
	print("a1", a1);
	print("e1", e1);
	print("p1a", p1);
	print("p1b", p1);

	linkedSpace.destroy();

	gpu.destroy();
	
}
