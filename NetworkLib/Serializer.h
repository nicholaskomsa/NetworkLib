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
		std::size_t mFileFrameCount = 0, mFileCurrentFrame = 0, mStreamFrameSize = 0;

		std::future<void> mReadFuture;

	public:

		~Serializer() {
			closeStream();
		}

		Serializer() = default;

		std::size_t getStreamFrameSize() const {
			return mStreamFrameSize;
		}
		void closeStream() {

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

		std::size_t getFramePosition(std::size_t y) const {

			const auto& [frameX, frameY] = mFrameRect.first;
			return (y + frameY) * mFrameWidth + frameX;
		}

		void writeToFile() {

			const auto& [frameW, frameH] = mFrameRect.second;

			mFile.open(mFileName, std::ios::app | std::ios::binary);

			auto begin = &mSourceFloatSpaceView.front();

			auto writeFrameLine = [&](auto y) {

				auto framePos = getFramePosition(y);
		
				const float* frameBegin = &mSourceFloatSpaceView.front() + framePos;
				auto lineSize = frameW;

				mFile.write(reinterpret_cast<const char*>(frameBegin), lineSize * sizeof(float));

				};

			auto writeLastFrameLine = [&](auto y) {
				//the floatspace may not be able to complete the last line if its too small
				//while the frameSize is framew * frameh, the float space is not necessarily that large,
				//if this is the case, complete the rest of the frame line with 0s
				constexpr auto floatSize = sizeof(float);

				auto framePos = getFramePosition(y);
				const float* frameBegin = &mSourceFloatSpaceView.front() + framePos;

				auto lineSize = frameW;

				if (framePos + lineSize >= mSourceFloatSpaceView.size()) {

					lineSize = mSourceFloatSpaceView.size() - framePos;

					mFile.write(reinterpret_cast<const char*>(frameBegin), lineSize * floatSize;

					char zero = 0;
					for (auto z : std::views::iota(lineSize, frameW))
						mFile.put(zero);
				}
				else
					mFile.write(reinterpret_cast<const char*>(frameBegin), lineSize * floatSize);

				};

			for (auto y : std::views::iota(0ULL, frameH-1 ))
				writeFrameLine(y);

			writeLastFrameLine(frameH-1);

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

				const auto& [frameW, frameH] = frameRect.second;

				constexpr auto floatSize = sizeof(float);

				auto gotoFramePosition = [&](std::size_t offset) {
					auto frameStart = floatSize + mFileCurrentFrame * mStreamFrameSize * floatSize;
					mFile.seekg(frameStart + offset * floatSize, std::ios::beg);
					};

				auto readFrameLine = [&](auto y) {

					auto framePos = getFramePosition(y);

					gotoFramePosition(framePos);

					auto& backBuffer = mBuffers.second;
					float* frameBegin = &backBuffer.front() + framePos;
					std::size_t lineSize = frameW;
					mFile.read(reinterpret_cast<char*>(frameBegin), lineSize * floatSize);

					};

				for (auto y : std::views::iota(0ULL, frameH))
					readFrameLine(y);

				++mFileCurrentFrame;

				}, mFrameRect, mFrameWidth);
		}

		void restartReading() {

			closeStream();

			mFile.open(mFileName, std::ios::in | std::ios::binary);

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