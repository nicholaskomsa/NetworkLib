#pragma once

#include <Gpt2.h>

#include <fstream>
#include <vector>
#include <future>
#include <optional>

#include "FloatSpaceConvert.h"

namespace NetworkLib {

	class Serializer {
	public:

		FloatSpaceConvert::FloatSpaceDimensions mFrameRect;
		std::size_t mFrameWidth = 0;

		using Frame = Tensor::Floats;
		using FrontAndBackFrame = std::pair<Frame, Frame>;
		FrontAndBackFrame mBuffers;

		Tensor::ConstView mSourceFloatSpaceView;

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
			auto& dimensions = mFrameRect.second;
			mFile.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));

			mFile.close();
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
				//the floatspace may not be able to complete the last line if its too small
				//while the frameSize is framew * frameh, the float space is not necessarily that large,
				//if this is the case, complete the rest of the frame line with 0s
				constexpr auto floatSize = sizeof(float);

				auto framePos = getFramePosition(y);
				const float* frameBegin = &mSourceFloatSpaceView.front() + framePos;

				auto lineSize = frameW;

				if (framePos + lineSize >= mSourceFloatSpaceView.size()) {

					lineSize = mSourceFloatSpaceView.size() - framePos;

					mFile.write(reinterpret_cast<const char*>(frameBegin), lineSize * floatSize);

					char zero = 0;
					for (auto z : std::views::iota(lineSize, frameW))
						mFile.put(zero);
				}
				else
					mFile.write(reinterpret_cast<const char*>(frameBegin), lineSize * floatSize);

				};

			for (auto y : std::views::iota(0ULL, frameH ))
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

			auto& frontBuffer = mBuffers.first;

			if (mFileCurrentFrame == 0)
				//the frame is loading still
				return std::nullopt;
			else
				return frontBuffer;
		}

	private:

		bool bufferReady() {
			return mReadFuture.valid() && mReadFuture.wait_for(0s) == std::future_status::ready;
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

				constexpr auto floatSize = sizeof(float);
				constexpr auto headerSize = sizeof(mFrameRect.second); // dimensions

				auto gotoFramePosition = [&](std::size_t offset) {
					auto frameStart = headerSize + mFileCurrentFrame * mStreamFrameSize * floatSize;
					mFile.seekg(frameStart + offset * floatSize, std::ios::beg);
					};

				const auto& [frameW, frameH] = frameRect.second;

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

			//read header

			auto& dimensions = mFrameRect.first;
			mFile.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));

			auto& [w, h] = dimensions;
			mStreamFrameSize = w * h;

			mFile.seekg(0, std::ios::end);
			auto animationBytesSize = mFile.tellg();
			animationBytesSize -= sizeof(dimensions);

			mFile.seekg(0, std::ios::beg);

			mFileCurrentFrame = 0;
			mFileFrameCount = animationBytesSize / sizeof(float) / mStreamFrameSize;

		}
	};
};