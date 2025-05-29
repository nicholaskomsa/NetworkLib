#include <SDL3/SDL.h>


int main() {
    // Initialize SDL3
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create a fullscreen window
    SDL_Window* window = SDL_CreateWindow("SDL3 Fullscreen Example",
                1920, 1080
        , SDL_WINDOW_BORDERLESS); //SDL_WINDOW_FULLSCREEN || SDL_WINDOW_OPENGL

    if (!window) {
        SDL_Log("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    // Main loop
    bool running = true;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
    }

    // Cleanup
    SDL_DestroyWindow(window);
    SDL_Quit();
   

	return 0;
}