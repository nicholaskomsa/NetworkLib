#include <SDL3/SDL.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <helper_gl.h> //NVIDA CUDA TOOLKIT

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <execution>

#include <FloatSpaceConvert.h>
#include <future>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd){

    auto [width, height] = FloatSpaceConvert::getDimensions(50000);

    SDL_GLContext glContext;
    GLuint texture;
    SDL_Window* window;

    std::vector<std::uint32_t> pixels(width * height);
    std::vector<float> floats(pixels.size());

    auto initOpenGL = [&]() {
        // Initialize SDL3
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            SDL_Log("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
            return -1;
        }

        // Create a fullscreen window
        window = SDL_CreateWindow("SDL3 Fullscreen Example",
            1920, 1080
            , SDL_WINDOW_MOUSE_FOCUS | SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_BORDERLESS 
            | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN );

        if (!window) {
            SDL_Log("Window could not be created! SDL_Error: %s\n", SDL_GetError());
            SDL_Quit();
            return -1;
        }

        glContext = SDL_GL_CreateContext(window);

        SDL_GL_SetSwapInterval(0); 

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glEnable(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA
            , GL_UNSIGNED_INT_8_8_8_8_REV, pixels.data());

    };

    auto render = [&]() {

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA
            , GL_UNSIGNED_INT_8_8_8_8_REV, pixels.data());

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glEnd();
  
        glFlush();
  
        SDL_GL_SwapWindow(window);
        };

    bool running = true;

    auto doEvents = [&]() {
        
        std::this_thread::yield();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
        };
    
    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);

    auto step = [&]() {

        std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
            return range(random);
            });

        FloatSpaceConvert::floatSpaceConvert(floats, pixels);

        };

    initOpenGL();

    std::size_t generation = 0;
    using namespace std::chrono;
    using namespace std::chrono_literals;

    constexpr std::chrono::milliseconds mLengthOfStep =  milliseconds(1s) / 10;

    nanoseconds lag(0), elapsedTime(0s);
    steady_clock::time_point nowTime, fpsStartTime
        , oldTime = steady_clock::now(), fpsEndTime = oldTime;

    std::uint32_t tickCount = 0;

    do {
        nowTime = high_resolution_clock::now();
        elapsedTime = duration_cast<nanoseconds>(nowTime - oldTime);
        oldTime = nowTime;

        lag += elapsedTime;
        while (lag >= mLengthOfStep) {

            step();

            render();

            lag -= mLengthOfStep;

            ++tickCount;
        }

        doEvents();


    } while (running && generation < 100 );

    SDL_DestroyWindow(window);
    SDL_Quit();

   
	return 0;
}