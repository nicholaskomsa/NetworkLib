
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <SDL3/SDL.h>
#include <helper_gl.h> //NVIDIA CUDA TOOLKIT

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <execution>
#include <functional>

#include <FloatSpaceConvert.h>

class Animator {
public:
    struct Error : public std::system_error {

        Error(std::errc code, const std::string& message)
            : std::system_error(int(code), std::generic_category(), message) {}

        static void sdlError() {
            throw Error(std::errc::operation_canceled, std::format("Graphics Error: {}", SDL_GetError() ));
        }
        void msgbox() const {
            MessageBoxA(nullptr, what(), "Animator Error", MB_OK | MB_ICONERROR);
        }
    };

private:

    std::size_t mWidth = 0 , mHeight = 0;

    using PixelsView = std::span<std::uint32_t>;
    std::vector<std::uint32_t> mPixels;


    SDL_GLContext mGLContext = nullptr;
    GLuint mTexture = 0;
    SDL_Window* mWindow = nullptr;

    bool mRunning = false;

public:
    void render() {

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA
            , GL_UNSIGNED_INT_8_8_8_8_REV, mPixels.data());

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glEnd();

        glFlush();

        SDL_GL_SwapWindow(mWindow);
    }
    void doEvents() {

        std::this_thread::yield();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                mRunning = false;
            }
        }
    }
    Animator(std::size_t width, std::size_t height) {

        mWidth = width;
        mHeight = height;

        mPixels.resize(width * height);

        auto initOpenGL = [&]() {
            // Initialize SDL3
            if (SDL_Init(SDL_INIT_VIDEO) < 0) 
                Error::sdlError();

            // Create a fullscreen window
            mWindow = SDL_CreateWindow("Float Space Animator",
                1920, 1080
                , SDL_WINDOW_MOUSE_FOCUS | SDL_WINDOW_INPUT_FOCUS
                | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN);

            if (!mWindow) 
                Error::sdlError();

            mGLContext = SDL_GL_CreateContext(mWindow);

            SDL_GL_SetSwapInterval(-1);

            glGenTextures(1, &mTexture);
            glBindTexture(GL_TEXTURE_2D, mTexture);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            glEnable(GL_TEXTURE_2D);

            glBindTexture(GL_TEXTURE_2D, mTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA
                , GL_UNSIGNED_INT_8_8_8_8_REV, mPixels.data());

            };

        initOpenGL();

    }
    ~Animator() {

        auto shutdownGL = [&]() {
            
            if( mWindow ) SDL_DestroyWindow(mWindow);
            SDL_Quit();

            mWindow = nullptr;
            };

        shutdownGL();
    }
    using StepFunction = std::function<void(PixelsView)>;
    void run(StepFunction&& step) {

        mRunning = true;

        using namespace std::chrono;
        using namespace std::chrono_literals;

        constexpr std::chrono::milliseconds mLengthOfStep = milliseconds(1s) / 10;

        nanoseconds lag(0), elapsedTime(0);
        steady_clock::time_point nowTime, oldTime = steady_clock::now() - mLengthOfStep;
        std::uint32_t tickCount = 0;

        do {
            nowTime = high_resolution_clock::now();
            elapsedTime = duration_cast<nanoseconds>(nowTime - oldTime);
            oldTime = nowTime;

            lag += elapsedTime;
            while (lag >= mLengthOfStep) {

                step(mPixels);

                render();

                lag -= mLengthOfStep;

                ++tickCount;
            }

            doEvents();

        } while (mRunning);
    }

    std::size_t getSize(){ return mWidth * mHeight; }
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd){
    
    try {
        auto [width, height] = FloatSpaceConvert::getDimensions(100000);

        Animator animator(width, height);

        std::mt19937 random;
        std::uniform_real_distribution<float> range(-1.0f, 1.0f);
        std::vector<float> floats(animator.getSize());

        auto step = [&](auto pixels) {

            std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
                return range(random);
                });

            FloatSpaceConvert::floatSpaceConvert(floats, pixels);

            };

        animator.run(step);
    }
    catch (const Animator::Error& e) {
        e.msgbox();
    }
	return 0;
}