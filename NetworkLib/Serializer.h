#pragma once

#include <Gpt2.h>

#include <fstream>
#include <vector>
#include <future>
#include <optional>

#include <FloatSpaceConvert.h>

namespace NetworkLib {

	class Serializer {
	
		FloatSpaceConvert::Rect mFrameRect;
		std::size_t mSourceWidth = 0;

		using Frame = Tensor::Floats;
		using FrontAndBackFrame = std::pair<Frame, Frame>;
		FrontAndBackFrame mBuffers;

		Tensor::ConstView mSourceFloatSpaceView;

		std::fstream mFile;
		std::string mFileName;
		std::size_t mFileFrameCount = 0, mFileCurrentFrame = 0
			, mStreamFrameSize = 0;

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

		void createOutputStream(NetworkLib::Tensor::ConstView floatSpaceView, const FloatSpaceConvert::Rect& subFrameRect
			, std::size_t frameWidth, const std::string_view fileName = "gpt2.animation") {

			mFileName = fileName;

			mSourceFloatSpaceView = floatSpaceView;
			mFrameRect = subFrameRect;
			mSourceWidth = frameWidth;

			mFile.open(mFileName, std::ios::out | std::ios::binary);

			//write header
			auto& dimensions = mFrameRect.mDimensions;
			mFile.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));

			mFile.close();
		}

		void writeToFile() {

			auto& origin = mFrameRect.mOrigin;
			auto& [frameW, frameH] = mFrameRect.mDimensions;

			mFile.open(mFileName, std::ios::app | std::ios::binary);

			auto writeFrameLine = [&](auto y) {
				//the floatspace may not be able to complete the last line if its too small
				//while the frameSize is framew * frameh, the float space is not necessarily that large,
				//if this is the case, complete the rest of the frame line with 0s
				constexpr auto floatSize = sizeof(float);

				auto frameLinePos = getFrameLinePosition(y, origin );
				std::size_t lineSize = 0, fillSize = 0;
				const float* lineBegin = nullptr;
				const auto& sourceSize = mSourceFloatSpaceView.size();

				if (frameLinePos >= sourceSize) {
					fillSize = frameW;
				} else {

					lineBegin = &mSourceFloatSpaceView.front() + frameLinePos;

					if (frameLinePos + frameW >= sourceSize) {

						lineSize = sourceSize - frameLinePos;
						fillSize = frameW - lineSize;
					}else
						lineSize = frameW;

					mFile.write(reinterpret_cast<const char*>(lineBegin), lineSize * floatSize);
				}

				if (fillSize) {

					char zero = 0;
					for (auto z : std::views::iota(0ULL, fillSize))
						mFile.put(zero);
				}

				};

			for (auto y : std::views::iota(0ULL, frameH ))
				writeFrameLine(y);

			mFile.close();
		}

		FloatSpaceConvert::Dimensions createInputStream(const std::string_view fileName = "gpt2.animation") {

			mFileName = fileName;

			restartReading();

			//images are of entire file float space
			mBuffers.first.resize(mStreamFrameSize);
			mBuffers.second.resize(mStreamFrameSize);

			return mFrameRect.mDimensions;
		}

		std::optional<NetworkLib::Tensor::View> getCurrentFrame(const FloatSpaceConvert::Rect& frameRect) {

			swapBuffers(frameRect);

			auto& frontBuffer = mBuffers.first;

			if (mFileCurrentFrame == 0)
				//the frame is loading still
				return std::nullopt;
			else
				return frontBuffer;
		}
		
		std::size_t getFrameLinePosition(std::size_t y, const FloatSpaceConvert::Coord& origin) {

			const auto& [frameX, frameY] = origin;
			return (y + frameY) * mSourceWidth + frameX;
		}
		
		void readBackBuffer(const FloatSpaceConvert::Rect& frameSubRect) {

			if (mFileFrameCount == mFileCurrentFrame)
				restartReading();

			mReadFuture = std::async(std::launch::async, [&](const auto frameSubRect) {

				constexpr auto floatSize = sizeof(float);
				auto& [ origin, dimensions ] = frameSubRect;

				auto gotoFrameLinePosition = [&](std::size_t frameLinePos) {
					constexpr auto headerSize = sizeof(dimensions);
					auto frameStart = headerSize + mFileCurrentFrame * mStreamFrameSize * floatSize;
					mFile.seekg(frameStart + frameLinePos * floatSize, std::ios::beg);
					};

				auto& [frameW, frameH] = dimensions;

				auto readFrameLine = [&](auto y) {

					auto frameLinePos = getFrameLinePosition(y, origin);
					gotoFrameLinePosition(frameLinePos);

					auto& backBuffer = mBuffers.second;
					float* lineBegin = &backBuffer.front() + frameLinePos;

					auto lineSize = frameW;

					mFile.read(reinterpret_cast<char*>(lineBegin), lineSize * floatSize);

					};

				for (auto y : std::views::iota(0ULL, frameH))
					readFrameLine(y);

				++mFileCurrentFrame;

				}, frameSubRect);
		}

	private:

		bool bufferReady() {
			return mReadFuture.valid() && mReadFuture.wait_for(0s) == std::future_status::ready;
		}
		void swapBuffers(const FloatSpaceConvert::Rect& subFrameRect) {

			if (bufferReady()) {

				std::swap(mBuffers.first, mBuffers.second);

				readBackBuffer(subFrameRect);
			}
		}

		void restartReading() {

			closeStream();

			mFile.open(mFileName, std::ios::in | std::ios::binary);

			mFrameRect = { {0,0}, {0,0} };
			auto& dimensions = mFrameRect.mDimensions;
			//read header
			mFile.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));

			auto& [w, h] = dimensions;
			mStreamFrameSize = w * h;
			mSourceWidth = w;

			mFile.seekg(0, std::ios::end);
			auto animationBytesSize = mFile.tellg();
			animationBytesSize -= sizeof(dimensions);

			mFile.seekg(0, std::ios::beg);

			mFileCurrentFrame = 0;
			mFileFrameCount = animationBytesSize / sizeof(float) / mStreamFrameSize;
		}
	};
};