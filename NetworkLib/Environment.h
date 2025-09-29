#pragma once

#include <atomic>

#include "GpuTensor.h"

namespace NetworkLib {

	namespace Gpu {

		class Environment {
		public:
			void create();
			void destroy();
			cublasHandle_t getBlas();
			cudaStream_t getStream();
			operator cudaStream_t();

			void vecScale(GpuView1& a1, float scale);
			void vecAddVec(const GpuView1& a1, GpuView1& o1);
			void matMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1);
			void batchedMatMulVec1(const GpuView2& w2, const GpuView2& i2, GpuView2& o2);
			void batchedMatMulVec(const GpuView2& w2, const GpuView2& i2, GpuView2& o2);
			void batchedMatTMulVec(const GpuView2& w2, const GpuView2& i2, GpuView2& o2);
			void matTMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1);

			void score(const GpuView2& sought, const GpuView2& desired);
			void resetMissesResult();
			int getMissesResult();

			void mse(const GpuView2& sought, const GpuView2& desired);
			float getMseResult();
			void resetMseResult();
			void downloadConvergenceResults();

			void relu(const GpuView1& o1, GpuView1& a1);
			void applyReluPrime(const GpuView1& a1, GpuView1& p1);
			void softmax(const GpuView1& o1, GpuView1& a1);
			void batchedSoftmax1(const GpuView2& o2, GpuView2& a2);
			void batchedSoftmax(const GpuView2& o2, GpuView2& a2);
			void diff(const GpuView1& desired1, const GpuView1& sought1, GpuView1& primes1);
			void updateWeights(const GpuView1& seen, GpuView2& weights, const GpuView1& primes, float learnRate);
			void copy(const GpuView1& source, GpuView1& dest);
			void batchedCopy(const GpuView2& source, GpuView2& dest);
			void batchedBroadcast(const GpuView1& source, GpuView2& dest);
			void batchedBroadcastAdd(const GpuView1& source, GpuView2& dest);
			void batchedDiff(const GpuView2& desired2, const GpuView2& sought2, GpuView2& primes2);
			void batchedUpdateWeights(const GpuView2& seen, GpuView2& weights, const GpuView2& primes, float learnRate);
			void activationFunction(LayerTemplate::ActivationFunction af, const GpuView1& o1, GpuView1& a1);
			void batchedActivationFunction(LayerTemplate::ActivationFunction af, const GpuView2& o2, GpuView2& a2);
			void activationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView1& a1, GpuView1& p1);
			void batchedActivationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView2& a2, GpuView2& p2);
			void errorFunction(LayerTemplate::ActivationFunction af, const GpuView1& desired, const GpuView1& sought, GpuView1& p1);
			void batchedErrorFunction(LayerTemplate::ActivationFunction af, const GpuView2& desired2, const GpuView2& sought2, GpuView2& p2);
		
			void sync();
			void deviceSync();
			void commandQueueSync(std::size_t commandCount=1);
			
			static void example();

		private:
			cublasHandle_t mHandle;
			cudaStream_t mStream;

			static constexpr std::size_t mMaxQueuedCommands = 50000;
			static std::atomic<std::size_t> mCommandCounter;

			Gpu::LinkedFloatSpace mLinkedFloatSpace;
			Float mMseResult;
			Int mMissesResult;
		};
	}
}