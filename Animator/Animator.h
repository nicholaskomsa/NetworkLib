#pragma once

#include <SDL3/SDL.h>
#include <GL/glew.h>
#include <freetype/freetype.h>

#include <FloatSpaceConvert.h>

#include <array>
#include <system_error>
#include <chrono>
#include <functional>
#include <algorithm>

#include "Geometry.h"
#include "Text.h"


using namespace std::chrono;
using namespace std::chrono_literals;

class Animator {
public:
    struct Error : public std::system_error {

        Error(std::errc code, const std::string& message);
        static void sdlError();
        static void glewError(auto error);
        static void glCompilationError(auto shaderProgram);

    };

    constexpr static std::size_t mWindowWidth = 1920, mWindowHeight = 1080;
    constexpr static float mAspectRatio = float(mWindowWidth) / mWindowHeight;
    constexpr static nanoseconds mLengthOfStep = nanoseconds(1s) / 7;

private:

    std::size_t mFrameWidth = 0, mFrameHeight = 0;

    using PixelsView = std::span<std::uint32_t>;
    std::vector<std::uint32_t> mPixels;
    using FloatsView = std::span<float>;
    FloatsView mFloats;

    using ColorizeMode = FloatSpaceConvert::ColorizeMode;
    ColorizeMode mColorizeMode = ColorizeMode::ROYGBIV;
    using Stripes = std::array<const std::size_t, 6>;
    Stripes mStripes = { 1,2,10,50,100,1000 };
    Stripes::iterator mSelectedStripes;
    float mX = 0.0f, mY = 0.0f, mTranslateSpeed = 0.1f, mScale = 1.0f;
    static constexpr milliseconds mKeyRepeatTime = 1000ms / 3;

    SDL_GLContext mGLContext = nullptr;
    SDL_Window* mWindow = nullptr;
    GLuint mViewerTexture = 0, mShaderProgram = 0;

    float mTextScale = 0.02;

    QuadManager mQuadManager;
    QuadManager::QuadReference mViewerQuadRef;
    TextManager mTextManager;
    TextManager::TextAreaReference mTextAreaRef;
    TextArea::LabeledValueReference mTicksValueRef, mColorModeValueRef, mStripeNumValueRef;

    bool mRunning = false, mPaused = false;

    void render();
    void doEvents();

    FloatSpaceConvert::Rect mFloatRect;

    void resizeTexture(){
        if(mViewerTexture)
            glDeleteTextures(1, &mViewerTexture);

        glGenTextures(1, &mViewerTexture);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, mViewerTexture);

        auto& [coord, dims] = mFloatRect;
        auto& [width, height] = dims;

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
        };
public:
    Animator() = default;

    Animator(std::size_t width, std::size_t height);
    ~Animator();

    std::function<void(void)> mCreateCustomGui, mCustomGuiRender;
    std::function<void(bool&, bool&)> mCustomGuiEvents;

    void setup(FloatsView floats);
    void shutdown();

    using StepFunction = std::function<bool(FloatsView)>;
    void run(StepFunction&& step);

    void animateMT19937(std::size_t floatCount=100000);
    void viewChatGPT2();
    void animateChatGPT2();
    void animateXORNetwork();

    void floatSpaceConvert() {

        if (mFloats.empty()) return;

        FloatSpaceConvert::floatSubSpaceConvert(mFloats, mPixels
            , mFloatRect, mFrameWidth
            , mColorizeMode, 0.0f, 1.0f, *mSelectedStripes);

        auto& textArea = mTextManager.getTextArea(mTextAreaRef);
        textArea.updateLabeledValue(mColorModeValueRef
            , std::format("{}x{}"
                , FloatSpaceConvert::getColorNames()[mColorizeMode]
                , *mSelectedStripes));
    }
    void setDimensions() {

        mFloatRect = FloatSpaceConvert::getFloatSpaceRect(mX, mY, mScale, mFrameWidth, mFrameHeight);

        static FloatSpaceConvert::Dimensions oldDimensions;
		const auto& dimensions = mFloatRect.mDimensions;

        if (oldDimensions !=  dimensions) {

            resizeTexture();
            const auto& [pw, ph] = dimensions;

            mPixels.resize(pw * ph);
            mPixels.shrink_to_fit();
            oldDimensions = dimensions;
        }
    }
  
    std::size_t getSize();
};
