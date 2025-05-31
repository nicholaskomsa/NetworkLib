
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <SDL3/SDL.h>

#include <GL/glew.h>
#include <GL/wglew.h>

#include <FloatSpaceConvert.h>

#include <format>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <execution>

class Animator {
public:
    struct Error : public std::system_error {

        Error(std::errc code, const std::string& message)
            : std::system_error(int(code), std::generic_category(), message) {}

        static void sdlError() {
            throw Error(std::errc::operation_canceled, std::format("SDL Error: {}", SDL_GetError()));
        }
        static void glewError(auto error) {
            throw Error(std::errc::operation_canceled, std::format("GLew Error: {}", reinterpret_cast<const char*>(error)));
        }

        static void glCompilationError(auto shaderProgram) {
            char infoLog[512];
            glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
            throw Error(std::errc::operation_canceled, std::format("GL Shader Compilation Error: {}", infoLog));
        }
        void msgbox() const {
            MessageBoxA(nullptr, what(), "Animator Error", MB_OK | MB_ICONERROR);
        }
    };

    constexpr static std::size_t mWindowWidth = 1920, mWindowHeight = 1080;
    constexpr static float mAspectRatio = float(mWindowWidth) / float(mWindowHeight);

private:

    std::size_t mWidth = 0, mHeight = 0;

    using PixelsView = std::span<std::uint32_t>;
    std::vector<std::uint32_t> mPixels;

    SDL_GLContext mGLContext = nullptr;
    GLuint mTexture = 0;
    SDL_Window* mWindow = nullptr;
    
    using GLBuffers = std::pair<GLuint, GLuint>; // Vertex Array Object, Vertex Buffer Object
    GLBuffers mGLBuffers;

    GLuint mShaderProgram = 0;

    bool mRunning = false;

public:
    void render() {

        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mPixels.data());

        glBindVertexArray(mGLBuffers.first);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        SDL_GL_SwapWindow(mWindow);
    }
    void doEvents() {

        std::this_thread::yield();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                mRunning = false;

            }
            else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_ESCAPE) {
                    mRunning = false;
                }
            }
        }
    }

    Animator(std::size_t width, std::size_t height) {

        mWidth = width;
        mHeight = height;

        mPixels.resize(width * height);

        auto initGL = [&]() {
            // Initialize SDL3
            if (SDL_Init(SDL_INIT_VIDEO) < 0)
                Error::sdlError();

            // Create a fullscreen window
            mWindow = SDL_CreateWindow("Float Space Animator",
                mWindowWidth, mWindowHeight
                , SDL_WINDOW_MOUSE_FOCUS | SDL_WINDOW_INPUT_FOCUS
                | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN);

            if (!mWindow)
                Error::sdlError();

            mGLContext = SDL_GL_CreateContext(mWindow);
            SDL_GL_SetSwapInterval(1);

            auto setupModernGL = [&]() {

                auto compileShader = [&](GLenum shaderType, const std::string& source) {
                    GLuint shader = glCreateShader(shaderType);
                    const char* src = source.c_str();
                    glShaderSource(shader, 1, &src, nullptr);
                    glCompileShader(shader);

                    // Check for compilation errors
                    GLint success;
                    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
                    if (!success)
                        Error::glCompilationError(shader);

                    return shader;
                    };

                auto createShaderProgram = [&](const std::string& vertexSrc, const std::string& fragmentSrc) {

                    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
                    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

                    GLuint shaderProgram = glCreateProgram();
                    glAttachShader(shaderProgram, vertexShader);
                    glAttachShader(shaderProgram, fragmentShader);
                    glLinkProgram(shaderProgram);

                    // Check for linking errors
                    GLint success;
                    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

                    // Cleanup shaders (they are now linked)
                    glDeleteShader(vertexShader);
                    glDeleteShader(fragmentShader);

                    if (!success)
                        Error::glCompilationError(shaderProgram);

                    return shaderProgram;
                    };

                auto getOrtho = [&](float left=-1, float right=1, float top=1, float bottom=-1, float n = -1.0f, float f = 1.0f) ->std::vector<float> {
                    return {
                        2.0f / (right - left),  0.0f,                   0.0f,                 0.0f,
                        0.0f,                   2.0f / (top - bottom),  0.0f,                 0.0f,
                        0.0f,                   0.0f,                   -2.0f / (f - n), 0.0f,
                        -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(f + n) / (f - n), 1.0f
                    };
                    };

                glewExperimental = GL_TRUE;
                GLenum err = glewInit();
                if (err != GLEW_OK)
                    Error::glewError(glewGetErrorString(err));

                std::string vertexShader = R"(
                    #version 330 core
                    layout(location = 0) in vec2 position;
                    layout(location = 1) in vec2 texCoord;

                    out vec2 fragTexCoord;

                    uniform mat4 projection;

                    void main() {
                        fragTexCoord = texCoord;
                        gl_Position = projection * vec4(position, 0.0, 1.0);
                    }

                    )"
                , fragmentShader = R"(
                    #version 330 core
                    in vec2 fragTexCoord;
                    out vec4 color;

                    uniform sampler2D textureSampler;

                    void main() {
                        color = texture(textureSampler, fragTexCoord);
                    }
                    )";

                mShaderProgram = createShaderProgram(vertexShader, fragmentShader);
                glUseProgram(mShaderProgram);

                GLint projLoc = glGetUniformLocation(mShaderProgram, "projection");
                glUniformMatrix4fv(projLoc, 1, GL_FALSE, getOrtho().data());
                };
            auto createQuad = [&]() {

                auto& [vao, vbo] = mGLBuffers;

                GLfloat quadVertices[] = {
                    // X, Y positions   // Texture Coords
                    -1.0f,  1.0f,       0.0f, 0.0f,  // Top-left
                     1.0f,  1.0f,       1, 0.0f,  // Top-right
                    -1.0f, -1.0f,       0.0f, 1,  // Bottom-left
                     1.0f, -1.0f,       1, 1   // Bottom-right
                };

                glGenVertexArrays(1, &vao);
                glGenBuffers(1, &vbo);

                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

                // Position Attribute
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
                glEnableVertexAttribArray(0);

                // Texture Coordinate Attribute
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
                glEnableVertexAttribArray(1);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindVertexArray(0);

                };
            auto createTexture = [&]() {

                glGenTextures(1, &mTexture);

                glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, mTexture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

                glBindTexture(GL_TEXTURE_2D, 0); // Unbind for safety
                };

            setupModernGL();
            createQuad();
            createTexture();
            };

        initGL();

    }
    ~Animator() {

        auto shutdownGL = [&]() {

            glDeleteTextures(1, &mTexture);

            glDeleteBuffers(1, &mGLBuffers.second); // Delete Vertex Buffer Object

            glDeleteVertexArrays(1, &mGLBuffers.first); // Delete Vertex Array Object

            glDeleteProgram(mShaderProgram);

            SDL_GL_DestroyContext(mGLContext);

            if (mWindow) SDL_DestroyWindow(mWindow);
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

        constexpr auto mLengthOfStep = nanoseconds(1s) / 10;

        nanoseconds lag(0), elapsedTime(0);
        steady_clock::time_point nowTime, oldTime = steady_clock::now();
        std::uint32_t tickCount = 0;

        do {
            nowTime = high_resolution_clock::now();
            elapsedTime = nowTime - oldTime;
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

    void animateStatic() {

        std::mt19937 random;
        std::uniform_real_distribution<float> range(-1.0f, 1.0f);
        std::vector<float> floats(getSize());

        auto step = [&](auto pixels) {

            std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
                return range(random);
                });

            FloatSpaceConvert::floatSpaceConvert(floats, pixels);

            };

        run(step);
    }

    std::size_t getSize() { return mPixels.size(); }
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {

    try {
        auto [width, height] = FloatSpaceConvert::getDimensions(100000, Animator::mAspectRatio);

        Animator animator(width, height);

        animator.animateStatic();

    }
    catch (const Animator::Error& e) {
        e.msgbox();
    }

    return 0;
}