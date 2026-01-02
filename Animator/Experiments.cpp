#include "Experiments.h"

#include <Gpt2.h>
#include <Serializer.h>
#include <ModelLogic.h>
#include <ModelMNIST.h>

#include <random>
#include <future>
#include <execution>
#include <algorithm>

void AnimateMT19937::run() {

    auto floatCount = 100'000;

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(floatCount, Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);
    std::vector<float> floats(getSize());

    setup(floats);

    auto step = [&](auto floats) {

        std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
            return range(random);
            });

        return true;
        };

    Animator::run(step);
}

void AnimatorViewChatGPT2::run() {

    auto gpt2 = std::make_unique<NetworkLib::GPT2>(); //gpt2 is large and offsourced to heap

    gpt2->setup();

    auto tensorSpace = gpt2->getForward().getTensorSpace();

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(tensorSpace.size(), Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    mPaused = true;

    setup(tensorSpace);

    auto step = [&](auto floats) {
        return false;
        };

    Animator::run(step);
}

void AnimatorChatGPT2::run() {

    NetworkLib::Serializer serializer;

    auto [frameWidth, frameHeight] = serializer.createInputStream();
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    mScale = 4.0f;
    setup({});

    serializer.readBackBuffer(mFloatRect);

    auto step = [&](auto floats) {

        auto frame = serializer.getCurrentFrame(mFloatRect);
        if (frame.has_value()) {
            mFloats = frame.value();
            return true;
        }
        else
            mPaused = true;

        return false;
        };


    Animator::run(step);
}

void AnimatorXOR::run() {

    NetworkLib::Model::XOR xorModel;
    xorModel.create();
    auto& [gpu, gpuNetwork] = xorModel.mTrainingManager.getGpuTask();

    std::size_t generation = 0;

    auto selectedView = gpuNetwork.mWeights;
    auto selectedFloatsView = NetworkLib::Cpu::Tensor::view(selectedView.mView);

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(selectedFloatsView.size(), Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    TextArea::LabeledValueReference mMseValueRef, mAccuracyValueRef;

    mCreateCustomGui = [&]() {

        auto& textArea = mTextManager.getTextArea(mTextAreaRef);

        mMseValueRef = textArea.addLabeledValue(mTextManager.mMinecraftFontFace, "Mse:", "    ");
        mAccuracyValueRef = textArea.addLabeledValue(mTextManager.mMinecraftFontFace, "Accuracy:", "000");
        };

    mCustomGuiEvents = [&](bool& changeDimensions, bool& doConvert) {

        };

    mCustomGuiRender = [&]() {

        auto& network = xorModel.getNetwork();
        float mse = network.mMse
            , misses = network.mMisses;

        auto& textArea = mTextManager.getTextArea(mTextAreaRef);
        textArea.updateLabeledValue(mMseValueRef, std::to_string(mse));

        auto sampleNum = xorModel.mBatchedSamplesView.size() * xorModel.mBatchSize;
        float accuracy = (sampleNum - misses) / float(sampleNum) * 100.0f;
        textArea.updateLabeledValue(mAccuracyValueRef, std::to_string(accuracy));
        };

    setup(selectedFloatsView);

    auto step = [&](auto floats) {

        xorModel.train();

        xorModel.calculateConvergence();

        return true;
        };

    Animator::run(step);

    xorModel.destroy();
}



void AnimatorMNIST::run() {

    NetworkLib::Model::MNIST mnistModel;
    mnistModel.create();

    using Clock = std::chrono::high_resolution_clock;

    std::future<void> convergenceFuture;
    constexpr auto convergencePeriod = 1s;
    auto convergenceTime = Clock::now() - convergencePeriod;

    auto periodicCalculateConvergence = [&]() {

        auto convergenceDone = [&]()->bool {
            return convergenceFuture.valid() && convergenceFuture.wait_for(0s) == std::future_status::ready;
            };

        auto now = Clock::now();
        auto elapsed = now - convergenceTime;
        if (elapsed >= convergencePeriod) {

            if (!convergenceFuture.valid() || convergenceDone())
                convergenceFuture = std::async(std::launch::async, [&]() {
                mnistModel.calculateConvergence();
                convergenceTime = Clock::now();
                    });
        }

        };

    auto& [gpu, gpuNetwork] = *mnistModel.mGpuTaskTrain;

    auto selectedView = gpuNetwork.mWeights;
    auto selectedFloatsView = NetworkLib::Cpu::Tensor::view(selectedView.mView);

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(selectedFloatsView.size(), Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    TextArea::LabeledValueReference mMseValueRef, mAccuracyValueRef;

    std::size_t trainingOffset = 0;

    mCreateCustomGui = [&]() {

        auto& textArea = mTextManager.getTextArea(mTextAreaRef);

        mMseValueRef = textArea.addLabeledValue(mTextManager.mMinecraftFontFace, "Mse:", "    ");
        mAccuracyValueRef = textArea.addLabeledValue(mTextManager.mMinecraftFontFace, "Test Accuracy:", "000");
        };

    mCustomGuiEvents = [&](bool& changeDimensions, bool& doConvert) {

        };

    mCustomGuiRender = [&]() {

        auto& network = mnistModel.getConvergenceNetwork();
        float mse = network.mMse
            , misses = network.mMisses
            , accuracy = network.mAccuracy;

        auto& textArea = mTextManager.getTextArea(mTextAreaRef);
        textArea.updateLabeledValue(mMseValueRef, std::to_string(mse));

        textArea.updateLabeledValue(mAccuracyValueRef, std::to_string(accuracy));
        };

    setup(selectedFloatsView);

    auto step = [&](auto floats) {

        mnistModel.train(10, trainingOffset);
        trainingOffset += 10;

        periodicCalculateConvergence();
        return true;
        };

    mPaused = true;
    Animator::run(step);

    if (convergenceFuture.valid())
        convergenceFuture.get();

    mnistModel.destroy();
}
