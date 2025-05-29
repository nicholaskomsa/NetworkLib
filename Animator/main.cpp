#include <SDL3/SDL.h>

#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <helper_gl.h> //NVIDA CUDA TOOLKIT

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include <Algorithms.h>
#include <FloatSpaceConvert.h>
#include <future>

int main() {

    auto [width, height] = FloatSpaceConvert::getDimensions(100000);


    SDL_GLContext glContext;
    GLuint texture;
    SDL_Window* window;
    std::vector<std::uint32_t> pixels(width * height);

    std::vector<float> floats(pixels.size());

    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);

    auto initOpenGL = [&]() {
        // Initialize SDL3
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            SDL_Log("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
            return -1;
        }

        // Create a fullscreen window
        window = SDL_CreateWindow("SDL3 Fullscreen Example",
            1920, 1080
            , SDL_WINDOW_BORDERLESS | SDL_WINDOW_OPENGL); //SDL_WINDOW_FULLSCREEN || 

        if (!window) {
            SDL_Log("Window could not be created! SDL_Error: %s\n", SDL_GetError());
            SDL_Quit();
            return -1;
        }

        glContext = SDL_GL_CreateContext(window);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glEnable(GL_TEXTURE_2D);


    };

    auto render = [&]() {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA
            , GL_UNSIGNED_INT_8_8_8_8_REV, pixels.data());

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glEnd();

        SDL_GL_SwapWindow(window);
        };

    bool running = true;

    auto doEvents = [&]() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
        };

    auto step = [&]() {

        std::generate(floats.begin(), floats.end(), [&]() {
            return range(random);
            });

        FloatSpaceConvert::floatSpaceConvert(floats, pixels);

        };


    auto animationFuture = std::async(std::launch::async, [&]() {

        initOpenGL();

        std::size_t generation = 0;
        std::chrono::milliseconds mLengthOfStep{ 100 };

        using namespace std::chrono;
        using namespace std::chrono_literals;

        nanoseconds lag(0), timeElapsed(0s);
        steady_clock::time_point nowTime, fpsStartTime
            , oldTime = steady_clock::now(), fpsEndTime = oldTime;

        std::uint32_t frameCount = 0, tickCount = 0;

        do {
            nowTime = high_resolution_clock::now();
            timeElapsed = duration_cast<nanoseconds>(nowTime - oldTime);
            oldTime = nowTime;

            lag += timeElapsed;
            while (lag >= mLengthOfStep) {

                step();

                lag -= mLengthOfStep;

                ++tickCount;
            }

            fpsStartTime = high_resolution_clock::now();

            doEvents();
            render();

            ++frameCount;

            nanoseconds fpsElapsedTime(duration_cast<nanoseconds>(fpsStartTime - fpsEndTime));
            if (fpsElapsedTime >= 1s) {

                fpsEndTime = fpsStartTime;

                //updateSecondStatus(tickCount, frameCount);
                std::cout << std::format("Ticks: {} FPS: {}", tickCount, frameCount);

                ++generation;

                frameCount = 0;
                tickCount = 0;
            }
        } while (running && generation < 30);

        SDL_DestroyWindow(window);
        SDL_Quit();

        });

    while( animationFuture.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready  ) {
        std::this_thread::yield();
	}


   
	return 0;
}