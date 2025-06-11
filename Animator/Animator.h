#pragma once

#include <SDL3/SDL.h>
#include <GL/glew.h>

#include <FloatSpaceConvert.h>

#include <array>
#include <system_error>
#include <chrono>
#include <functional>

using namespace std::chrono;
using namespace std::chrono_literals;

class Animator {
public:
    struct Error : public std::system_error {

        Error(std::errc code, const std::string& message);
        static void sdlError();
        static void glewError(auto error);
        static void glCompilationError(auto shaderProgram);
        void msgbox() const;
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

    bool mRunning = false;

    void render();
    void doEvents();
    void updateCamera();

public:
    Animator() = default;

    Animator(std::size_t width, std::size_t height);
    ~Animator();

    void setup(FloatsView floats);
    void shutdown();

    using StepFunction = std::function<bool(FloatsView)>;
    void run(StepFunction&& step);

    void animateStatic(std::size_t floatCount=100000);
    void viewChatGPT2();

    void floatSpaceConvert() {
        FloatSpaceConvert::floatSubSpaceConvert(mFloats, mPixels
            , 0, 0, mTextureWidth, mTextureHeight
            , mColorizeMode, 0.0f, 1.0f, *mSelectedStripes);
    }
    std::size_t getSize();
};
