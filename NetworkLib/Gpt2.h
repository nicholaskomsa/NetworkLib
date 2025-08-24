#pragma once

#include <array>
#include <iostream>
#include <print>
#include <numbers>
#include <numeric>

#include <boost/bimap.hpp>

#include "Algorithms.h"
#include "Gpt2Tensor.h"
#include "Parallel.h";

namespace NetworkLib {

	class GPT2 {
	public:
		
		//configuration chat gpt2 model here
		static constexpr auto mFilePath = "./";
		static constexpr std::size_t mDVocab = 50257
			, mDModel = 768
			, mDSeq = 1024
			, mHeadNum = 12, mAttnLayersNum = 12
			, mHeadsPerDModel = mDModel / mHeadNum
			, mQOffset = 0, mKOffset = mDModel, mVOffset = mDModel * 2
			, mTestInputSize = mDSeq;	//vs dSeq for full size or 64 for test size

		using Floats = Tensor::Floats;
		using Token = std::uint16_t;
		static constexpr Token InvalidToken = std::numeric_limits<Token>::max();
		using Tokens = std::vector<Token>;
		using TokensView = std::span<Token>;
		using IotaView = std::ranges::iota_view<std::size_t, std::size_t>;

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message);

			static void fileNotFound(std::string_view fileName);
			static void wordNotFound(std::string_view word);
			static void tokenNotFound(Token token);
		};

		class Translator {
		public:
			using Word = std::string_view;
			using WordMap = boost::bimap<Word, Token>;

			void load();
			std::string decode(TokensView tokens) const;
			std::string decode(Token token) const;

			Tokens encode(std::string_view remaining) const;
			Token getToken(std::string_view word) const;
			Word getWord(Token token) const;
		private:

			WordMap mWordMap;
			std::string mDenseWords;

		} mTranslator;

		struct TestData {

			Tokens mTokens;

			void load();

		} mTestData;

		class Forward;
		class Backward;

		GPT2() = default;


		void setup();
		void chat();
		void slide(Tokens& tokens, std::size_t distance = 50);
		Forward& getForward() {
			return mForward;
		}
		class Diagnostics {

			using TestFunction = std::function<void(GPT2& gpt2)>;
			void run(TestFunction&& test);

		public:

			static double sumf(Tensor::ConstView tensorView, std::string_view expected);
			static double sumf(const Tensor& tensor, std::string_view expected);
			static double sumAbsf(Tensor::ConstView tensor, std::string_view expected);
			static double sumAbsf(const Tensor& tensor, std::string_view expected);
			static double attnSumAbsf(const Tensor& tensor, std::size_t offset, std::string_view expected);
		
			void firstCitizenTest64();
			void feedForwardSpeed1024();
			void simpleChat();
			void crossEntropyTest64();
			void backwardTest64();
			void SGDTest();
			void serializeTest();

		};
		friend class Diagnostics;

	private:

		static constexpr auto  mDModel3 = mDModel * 3
			, mDModel4 = mDModel * 4
			, mSeqModel = mDSeq * mDModel
			, mSeqModel3 = mDSeq * mDModel3
			, mSeqModel4 = mDSeq * mDModel4
			, mSeqSeqHead = mDSeq * mDSeq * mHeadNum
			, mSeqVocab = mDSeq * mDVocab
			, mModel4Model = mDModel4 * mDModel
			, mModel3Model = mDModel3 * mDModel
			, mModelModel = mDModel * mDModel
			, mVocabModel = mDVocab * mDModel;

		static TimeAverage<milliseconds> mForwardTime, mBackwardTime;

		static void forward(std::size_t i, const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void forward(const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void softmax(std::size_t i, Tensor::ConstView input, Tensor::View output);

		static void backward(const Tensor& dOutputs, const Tensor& weights, Tensor& dWeights, Tensor& dBias, const Tensor& inActivations, Tensor& outActivations, Parallel& parallel);
		static void softmaxBack(const IotaView& iotaView, Tensor::ConstView input, Tensor::ConstView output, Tensor::View dSoftmax);
		static void sgd(Tensor::View weights, Tensor::ConstView gradients, float learnRate = 0.0002);

		class MLP {

			friend class Diagnostics;

			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;

			Tensor mGeluCDF;

			static constexpr float r_sqrt2 = 1.0f / std::numbers::sqrt2;
			static const float r_sqrt2Pi;
		public:

			static std::size_t getBackwardSize();
			void load(Floats::iterator& backwardSpace);
			void load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace);
			const Tensor& getCProjActivations() const;

			void forward(const Tensor& input, Parallel& parallel);
			void forward(std::size_t i, const Tensor& input, Parallel& parallel);

			void backward(const MLP& mlp, const Tensor& linear, const Tensor& dResidual, Tensor& dLinear, Parallel& parallel);
			void sgd(const MLP& gradients, float learnRate);

			static TimeAverage<milliseconds> mBackwardGeluTime;
		};

		class LinearLayer {
		public:
			friend class Diagnostics;

			Tensor mBias, mWeight;
			Tensor mActivations;

			//for backward
			Floats mMean, mRStdDev;

			static TimeAverage<milliseconds> mBackwardTime;

		public:

			static std::size_t getBackwardSize();
			void load(Floats::iterator& backwardSpace);

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace);
			Tensor& getActivations();

			void normalise(std::size_t i, const Tensor& input);
			void normalise(const Tensor& input, Parallel& parallel);

			void backward(const LinearLayer& inputLayer, const Tensor& inputs, Tensor& dInputs, Parallel& parallel);
			void sgd(const LinearLayer& gradients, float learnRate);

		};
		class AttnLayer {

			friend class Diagnostics;

			static Parallel mParallelHeads;

			LinearLayer mL1, mL2;

			Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
			Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;
			Tensor mResidualActivation1, mResidualActivation2
				, mResidualActivation1Out; //for backward

			MLP mMLP;


			static const float r_sqrtHeadsPerDModel;

			void calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::View attnOut);
			void calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::ConstView attnOutSoftmax);
			
			void backwardVAtten(const AttnLayer& attn, std::size_t headOffset, const IotaView& qIotaView, std::size_t i, Tensor::ConstView inputAttnOutSoftmax, Tensor::View outputAttnOutSoftmax);
			void backwardQKAtten(const AttnLayer& attn, std::size_t headOffset, const IotaView& qIotaView, std::size_t i, Tensor::ConstView attnActivations);
			void multiHeadedAttn(std::size_t m);

			void attention(std::size_t m);
			void attention(Parallel& parallel);

			void multiHeadedAttnBack(AttnLayer& attn, Parallel& parallel);

			void residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor);
			void residual(const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor, Parallel& parallel);
			void residualBack(const Tensor& a, const Tensor& b, Tensor& outputTensor);

		public:

			using ReadTensorFunctor = std::function<Tensor(std::string_view)>;
			void load(ReadTensorFunctor&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace);

			static std::size_t getBackwardSize();
			void load(Floats::iterator& backwardSpace);

			Tensor& forward(const Tensor& inputTensor, Parallel& parallel);
			Tensor& forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel);

			Tensor& getOutput();
		
			void backward(AttnLayer& attn, const Tensor& forwardResidual2, Tensor& residual2, Parallel& parallel);
			void sgd(const AttnLayer& gradients, float learnRate);

			static TimeAverage<milliseconds> mForwardAttnTime, mBackwardAttnTime;

		};

		class Backward;

		class Forward {

			friend class Diagnostics;
			friend class Backward;

			Parallel mParallelInput, mParallelI;

			Floats mTensorSpace, mActivationSpace;

			Tensor mWpeWeight, mWteWeight, mWActivations, mUnembedActivations, mUnembedActivationsSoftmax;

			std::array<AttnLayer, mAttnLayersNum> mAttnLayers;
			LinearLayer mFinalLayer;

			void load();
			void embedInput(std::size_t i, Token token);
			void embedInputs(TokensView tokens);
			void unEmbedOutput(std::size_t i);
			void unEmbedOutputs();

		public:

			void setup();

			Token feedForward(TokensView tokens);
			Token feedMore(TokensView tokens);

			Token getPrediction(std::size_t i) const;
			float crossEntropyLoss(TokensView nextTokens);

			static TimeAverage<milliseconds> mUnembedTime, mLayersTime;

			static std::string getTimeAverages() {
				return std::format("{},{},{},{}",
					mUnembedTime.getString(), mLayersTime.getString()
					, mForwardTime.getString(), AttnLayer::mForwardAttnTime.getString() );
			}
			Tensor::View getTensorSpace() {
				return mTensorSpace;
			}
			Tensor::View getActivationSpace() {
				return mActivationSpace;
			}
		} mForward;

		class Backward {

			friend class Diagnostics;

			Parallel mParallelInput, mParallelUnembed;

			Floats mBackwardSpace;

			Tensor mUnembed, mWteWeight, mEmbed, mWpeWeight;

			LinearLayer mFinalLayer;

			std::array<AttnLayer, mAttnLayersNum> mAttnLayers;

			Forward* mForward = nullptr;

			static TimeAverage<milliseconds> mUnembedTime, mLayersTime;

		public:

			void setup(Forward* forward);

			void unEmbedOutputs(TokensView nextTokens);
			void embedOutputs(TokensView tokens, Parallel& parallel);
			void backward(TokensView tokens, TokensView nextTokens);
			void sgd(float learnRate = 0.0002);

			static std::string getTimeAverages() {
				return std::format("{},{},{},{}"
					, mUnembedTime.getString(), mLayersTime.getString()
					, mBackwardTime.getString(), AttnLayer::mBackwardAttnTime.getString());
			}

		} mBackward;


	};
}