#pragma once

#include <Gpt2.h>

#include <fstream>
#include <deque>
#include <mutex>
#include <vector>
#include <future>
#include <optional>

#include "FloatSpaceConvert.h"

namespace NetworkLib {
	class Serializer {
	public:

		FloatSpaceConvert::FloatSpaceDimensions mFrameRect;
		std::size_t mFrameWidth = 0;

		using Frame = NetworkLib::Tensor::Floats;
		using FrontAndBackFrame = std::pair<Frame, Frame>;
		FrontAndBackFrame mBuffers;

		NetworkLib::Tensor::ConstView mSourceFloatSpaceView;

		std::fstream mFile;
		std::string mFileName;
		std::size_t mFileFrameCount = 0, mFileCurrentFrame = 0;

		std::future<void> mReadFuture;
		bool mStopReading = false;
		std::size_t mFramesAvailable = 0;

		std::size_t mStreamFrameSize = 0, mTensorFrameSize = 0;

	public:

		~Serializer() {
			closeStream();
		}

		Serializer() = default;

		std::size_t getStreamFrameSize() const {
			return mStreamFrameSize;
		}
		void closeStream() {

			mStopReading = true;

			if (mFile.is_open())
				mFile.close();
		}

		void createOutputStream(NetworkLib::Tensor::ConstView floatSpaceView, FloatSpaceConvert::FloatSpaceDimensions frameRect, std::size_t frameWidth, const std::string_view fileName = "gpt2.animation") {

			mFileName = fileName;

			mSourceFloatSpaceView = floatSpaceView;
			mFrameRect = frameRect;
			mFrameWidth = frameWidth;

			mFile.open(mFileName, std::ios::out | std::ios::binary);

			//write header
			auto [w, h] = mFrameRect.second;
			std::size_t floatCount = w * h;

			mFile.write(reinterpret_cast<const char*>(&floatCount), sizeof(floatCount));
			mFile.close();
			mStreamFrameSize = floatCount;
		}

		void writeToFile() {

			const auto& [frameX, frameY] = mFrameRect.first;
			const auto& [frameW, frameH] = mFrameRect.second;

			mFile.open(mFileName, std::ios::app | std::ios::binary);

			auto begin = &mSourceFloatSpaceView.front();

			auto getFramePosition = [&](auto y) {
				return (y + frameY) * mFrameWidth + frameX;
				};

			std::size_t lineSize = frameW
				, completeLineSize = mFrameWidth;

			auto writeFrameLine = [&](auto y) {

				auto offset = getFramePosition(y);

				//no not read past eoframe
				//if (offset + lineSize > mStreamFrameSize)
				//	lineSize = mStreamFrameSize - (offset + lineSize);

				const float* frameBegin = &mSourceFloatSpaceView.front() + getFramePosition(y);
				mFile.write(reinterpret_cast<const char*>(frameBegin), lineSize * sizeof(float));

				};

			std::size_t y;
			for (y = 0; y < frameH - 1; ++y) {
				writeFrameLine(y);
			}
			writeFrameLine(y);

			mFile.close();
		}

		void createInputStream(const std::string_view fileName = "gpt2.animation") {

			mFileName = fileName;

			restartReading();
		}
		void startReadingWindow(FloatSpaceConvert::FloatSpaceDimensions frameRect, std::size_t frameWidth) {

			mFrameRect = frameRect;
			mFrameWidth = frameWidth;

			//images are of entire file float space
			mBuffers.first.resize(mStreamFrameSize);
			mBuffers.second.resize(mStreamFrameSize);

			readBackBuffer();
		}

		std::optional<NetworkLib::Tensor::View> getCurrentFrame() {

			swapBuffers();

			if (mFileCurrentFrame > 0)
				return mBuffers.first;
			else
				return std::nullopt;
		}

	private:

		bool bufferReady() {
			return mReadFuture.valid() && mReadFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
		}
		void swapBuffers() {

			if (bufferReady()) {

				std::swap(mBuffers.first, mBuffers.second);

				readBackBuffer();
			}
		}
		void readBackBuffer() {

			if (mFileFrameCount == mFileCurrentFrame)
				restartReading();

			mReadFuture = std::async(std::launch::async, [&](FloatSpaceConvert::FloatSpaceDimensions frameRect, std::size_t frameWidth) {

				const auto& [frameX, frameY] = frameRect.first;
				const auto& [frameW, frameH] = frameRect.second;

				constexpr auto floatSize = sizeof(float);

				auto gotoFrameOffset = [&](std::size_t offset) {
					auto frameStart = floatSize + mFileCurrentFrame * mStreamFrameSize * floatSize;
					mFile.seekg(frameStart + offset * floatSize, std::ios::beg);
					};

				auto getFramePosition = [&](auto y) {
					return (y + frameY) * frameWidth + frameX;
						};

				std::size_t lineSize = frameW;

				auto readFrameLine = [&](auto y) {

					auto offset = getFramePosition(y);

					gotoFrameOffset(offset);

					//no not read past eoframe
					if (offset + lineSize > mStreamFrameSize)
						lineSize = mStreamFrameSize - (offset + lineSize);

					float* frameBegin = &mBuffers.second.front() + getFramePosition(y);
					mFile.read(reinterpret_cast<char*>(frameBegin), lineSize * floatSize);

					};

				std::size_t y;
				for (y = 0; y < frameH - 1; ++y) {
					readFrameLine(y);
				}
				readFrameLine(y);

				++mFileCurrentFrame;

				}, mFrameRect, mFrameWidth);
		}

		void restartReading() {

			closeStream();

			mFile.open(mFileName, std::ios::in | std::ios::binary);
			mStopReading = false;

			//write header
			std::size_t floatCount = 0;
			mFile.read(reinterpret_cast<char*>(&floatCount), sizeof(floatCount));
			mStreamFrameSize = floatCount;

			mFile.seekg(0, std::ios::end);
			auto animationBytesSize = mFile.tellg();
			animationBytesSize -= sizeof(floatCount);

			mFile.seekg(0, std::ios::beg);

			mFileCurrentFrame = 0;
			mFileFrameCount = animationBytesSize / sizeof(float) / mStreamFrameSize;

		}
	};
};