#pragma once

#include <SDL3/SDL.h>
#include <GL/glew.h>

#include <FloatSpaceConvert.h>

#include <array>
#include <system_error>
#include <chrono>
#include <functional>
#include <algorithm>

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

    std::size_t mTextureWidth = 0, mTextureHeight = 0;

    using PixelsView = std::span<std::uint32_t>;
    std::vector<std::uint32_t> mPixels;
    using FloatsView = std::span<float>;
    FloatsView mFloats;

    using ColorizeMode = FloatSpaceConvert::ColorizeMode;
    ColorizeMode mColorizeMode = ColorizeMode::ROYGBIV;
    using Stripes = std::array<const std::size_t, 5>;
    Stripes mStripes = { 1,2,10,50,100 };
    Stripes::iterator mSelectedStripes;
    float mX = 0.0f, mY = 0.0f, mTranslateSpeed = 0.1f, mScale = 1.0f;
    static constexpr milliseconds mKeyRepeatTime = 1000ms / 3;

    SDL_GLContext mGLContext = nullptr;
    SDL_Window* mWindow = nullptr;
    GLuint mTexture = 0, mShaderProgram = 0, mVao = 0, mVbo = 0;

    bool mRunning = false, mPaused = false;

    void render();
    void doEvents();
    void updateCamera();


    using FloatSpaceCoord = std::pair<std::size_t,std::size_t>;
    using FloatSpaceDimensions = std::pair<FloatSpaceCoord, FloatSpaceCoord>;

    FloatSpaceDimensions mFloatSubSpaceDimensions;

    void resizeTexture(){
        if( mTexture)
            glDeleteTextures(1, &mTexture);

        glGenTextures(1, &mTexture);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, mTexture);

        auto& [coord, dims] = mFloatSubSpaceDimensions;
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

    void setup(FloatsView floats);
    void shutdown();

    using StepFunction = std::function<void(FloatsView)>;
    void run(StepFunction&& step);

    void animateMT19937(std::size_t floatCount=100000);
    void viewChatGPT2();

    void setFloatSubSpaceDimensions() {

        if( mScale < 1.0f ) mScale = 1.0f;

        float rScale = 1.0f / mScale;

        mX = std::clamp(mX, -1.0f + rScale, 1.0f - rScale);
        mY = std::clamp(mY, -1.0f + rScale, 1.0f - rScale);

        float x1 = mX - rScale, x2 = mX + rScale
            , y1 = mY + rScale, y2 = mY - rScale;

        x1 = (x1 + 1.0f) / 2.0f;
        x2 = (x2 + 1.0f) / 2.0f;

        y1 = (-y1 + 1.0f) / 2.0f;
        y2 = (-y2 + 1.0f) / 2.0f;

        x1 = std::clamp(x1, 0.0f, 1.0f);
        x2 = std::clamp(x2, 0.0f, 1.0f);
        y1 = std::clamp(y1, 0.0f, 1.0f);
        y2 = std::clamp(y2, 0.0f, 1.0f);

        std::size_t px1 = std::floor(x1 * mTextureWidth)
            , px2 = std::floor(x2 * mTextureWidth)
            , py1 = std::ceil(y1 * mTextureHeight)
            , py2 = std::ceil(y2 * mTextureHeight);

        static std::size_t oldW=0, oldH = 0;

        std::size_t pw = px2 - px1
            , ph = py2 - py1;

        mFloatSubSpaceDimensions = { {px1, py1}, {pw, ph} };

        if (oldW != pw || oldH != ph) {
            resizeTexture();

            mPixels.resize(pw * ph);
            mPixels.shrink_to_fit();
            oldW = pw;
            oldH = ph;
        }
    }
    
    void floatSpaceConvert() {

        auto [coord, dims] = mFloatSubSpaceDimensions;
  
        FloatSpaceConvert::floatSubSpaceConvert(mFloats, mPixels
            , coord.first, coord.second, dims.first, dims.second, mTextureWidth
            , mColorizeMode, 0.0f, 1.0f, *mSelectedStripes);
    }
    std::size_t getSize();
};
