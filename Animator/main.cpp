#include <SDL3/SDL.h>

#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <helper_gl.h> //NVIDA CUDA TOOLKIT

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include <FloatSpaceConvert.h>

int main() {

    auto [width, height] = FloatSpaceConvert::getDimensions(100000);

    std::vector<std::uint32_t> pixels(width * height); 

    std::vector<float> floats(pixels.size());

    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);
    std::generate(floats.begin(), floats.end(), [&]() {
        return range(random);
	});

    FloatSpaceConvert::floatSpaceConvert(floats, pixels);

    SDL_GLContext glContext;
    GLuint texture;
    SDL_Window* window;

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

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA
            , GL_UNSIGNED_INT_8_8_8_8_REV, pixels.data());
    };
    initOpenGL();

    bool running = true;
    SDL_Event event;
    std::size_t generation = 0;

    glBindTexture(GL_TEXTURE_2D, texture);

    while (running) {

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
  
       
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glEnd();

        SDL_GL_SwapWindow(window);
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
   
	return 0;
}