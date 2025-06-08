#include "Animator.h"

#include <format>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <execution>

Animator::Error::Error(std::errc code, const std::string& message)
    : std::system_error(int(code), std::generic_category(), message) {}

void Animator::Error::sdlError() {
    throw Error(std::errc::operation_canceled, std::format("SDL Error: {}", SDL_GetError()));
}
void Animator::Error::glewError(auto error) {

    throw Error(std::errc::operation_canceled, std::format("GLew Error: {}", reinterpret_cast<const char*>(glewGetErrorString(error))));
}
void Animator::Error::glCompilationError(auto shaderProgram) {
    char infoLog[512];
    glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
    throw Error(std::errc::operation_canceled, std::format("GL Shader Compilation Error: {}", infoLog));
}
void Animator::Error::msgbox() const {
    MessageBoxA(nullptr, what(), "Animator Error", MB_OK | MB_ICONERROR);
}
   
void Animator::render() {

    glBindTexture(GL_TEXTURE_2D, mTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mTextureWidth, mTextureHeight, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mPixels.data());
    glBindVertexArray(mVao);
   
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    SDL_GL_SwapWindow(mWindow);
}
void Animator::doEvents() {

    std::this_thread::yield();

    SDL_Event event;
    const auto& type = event.type;
    const auto& key = event.key.key;

    auto keyRepeatGuard = [&](auto&& keyRepeatCode) {

        //do keys at most sometimes
        auto nowTime = steady_clock::now();
        static auto endTime = nowTime - mKeyRepeatTime;
        static auto oldKey = SDLK_UNKNOWN;
        auto elapsed = nowTime - endTime;
       
        if (oldKey == key && elapsed < mKeyRepeatTime) return;

        oldKey = key;
        endTime = nowTime;

        keyRepeatCode();

        };

    auto doQuit = [&]() {
        return type == SDL_EVENT_QUIT
            || type == SDL_EVENT_KEY_DOWN && key == SDLK_ESCAPE;
        };
    auto keydown = [&]() {
        return type == SDL_EVENT_KEY_DOWN;
        };

    auto doUpdateQuad = [&]() {

        return key == SDLK_LEFT
            || key == SDLK_RIGHT
            || key == SDLK_UP
            || key == SDLK_DOWN
            || key == SDLK_S
            || key == SDLK_A
            || key == SDLK_R;
		};

    while (SDL_PollEvent(&event)) {

        if (doQuit()) {

            mRunning = false;
            return;
        
        }else if(keydown()) {

            keyRepeatGuard([&]() {

                switch (key) {
                case SDLK_1: mColorizeMode = ColorizeMode::NICKRGB; break;
                case SDLK_2: mColorizeMode = ColorizeMode::SHORTNRGB; break;
                case SDLK_3: mColorizeMode = ColorizeMode::ROYGBIV; break;
                case SDLK_4: mColorizeMode = ColorizeMode::GREYSCALE; break;
                case SDLK_5: mColorizeMode = ColorizeMode::BINARY; break;

                case SDLK_Q:
                    if (mSelectedStripes == mStripes.begin())
                        mSelectedStripes = std::prev(mStripes.end(), 1);
                    else
                        mSelectedStripes = std::prev(mSelectedStripes, 1);
                    break;
                case SDLK_W:
                    std::advance(mSelectedStripes, 1);
                    if (mSelectedStripes == mStripes.end())
                        mSelectedStripes = mStripes.begin();
                    break;

                case SDLK_LEFT:
                    mX += mTranslateSpeed;
                    break;
                case SDLK_RIGHT:
                    mX -= mTranslateSpeed;
                    break;
                case SDLK_UP:
                    mY -= mTranslateSpeed;
                    break;
                case SDLK_DOWN:
                    mY += mTranslateSpeed;
                    break;
                case SDLK_A:
                    mScale /= 2.0f;
                    break;
                case SDLK_S:
                    mScale *= 2.0f;
                    break;
                case SDLK_R:
                    mX = 0.0f;
                    mY = 0.0f;
                    mScale = 1.0f;
                    break;
                }

                if (doUpdateQuad())
                    updateQuad();

                });

        }
    }
}
Animator::Animator(std::size_t width, std::size_t height) {
    mTextureWidth = width;
    mTextureHeight = height;
}
Animator::~Animator() {
    shutdown();
}

void Animator::updateQuad(bool generate) {

    //scale and translate == sat
    auto sat = [&](float f, float t) {
        return (f + t) * mScale;
        };
    auto satx = [&](float x) {
        return sat(x, mX);
        };
    auto saty = [&](float y) {
        return sat(y, mY);
        };

    std::array<float, 16> vertices = {
        //vertex = X, Y, U, V
        satx(-1.0f), saty(1.0f),    0.0f, 0.0f  // Top-left
        , satx(1.0f), saty(1.0f),     1.0f, 0.0f  // Top-right
        , satx(-1.0f), saty(-1.0f),   0.0f, 1.0f  // Bottom-left
        , satx(1.0f), saty(-1.0f),    1.0f, 1.0f   // Bottom-right
    };

    if (generate) {
        glGenVertexArrays(1, &mVao);
        glGenBuffers(1, &mVbo);

        glBindVertexArray(mVao);
        glBindBuffer(GL_ARRAY_BUFFER, mVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

        // Position Attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
        glEnableVertexAttribArray(0);

        // Texture Coordinate Attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

    }else{
        glBindBuffer(GL_ARRAY_BUFFER, mVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * vertices.size(), vertices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}
void Animator::setup(FloatsView floats) {

    mFloats = floats;

    mPixels.resize(mTextureWidth * mTextureHeight, 0.0f);

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

                GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc)
                    , fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc)
                    , shaderProgram = glCreateProgram();

                glAttachShader(shaderProgram, vertexShader);
                glAttachShader(shaderProgram, fragmentShader);
                glLinkProgram(shaderProgram);

                // Cleanup shaders (they are now linked)
                glDeleteShader(vertexShader);
                glDeleteShader(fragmentShader);

                // Check for linking errors
                GLint success;
                glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

                if (!success)
                    Error::glCompilationError(shaderProgram);

                return shaderProgram;
                };

            glewExperimental = GL_TRUE;
            GLenum err = glewInit();
            if (err != GLEW_OK)
                Error::glewError(err);

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

            auto setOrthoProjection = [&](float left=-1, float right=1
                , float top=1, float bottom=-1
                , float n=-1, float f=1) {

                std::array<float, 16> ortho = {
                    2.0f / (right - left),  0.0f,                   0.0f,                 0.0f,
                    0.0f,                   2.0f / (top - bottom),  0.0f,                 0.0f,
                    0.0f,                   0.0f,                   -2.0f / (f - n),    0.0f,
                    -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(f + n) / (f - n), 1.0f
                    };

                GLint projLoc = glGetUniformLocation(mShaderProgram, "projection");

                glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho.data());
                };

            setOrthoProjection();
            };

        auto createTexture = [&]() {

            glGenTextures(1, &mTexture);

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, mTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mTextureWidth, mTextureHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            glBindTexture(GL_TEXTURE_2D, 0);
            };

        setupModernGL();
        createTexture();

        updateQuad(true);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        };

    initGL();

    mSelectedStripes = mStripes.begin();
}
void Animator::shutdown() {

    auto shutdownGL = [&]() {

        glDeleteTextures(1, &mTexture);

        glDeleteBuffers(1, &mVbo); // Delete Vertex Buffer Object

        glDeleteVertexArrays(1, &mVao); // Delete Vertex Array Object

        glDeleteProgram(mShaderProgram);

        SDL_GL_DestroyContext(mGLContext);

        SDL_DestroyWindow(mWindow);
        SDL_Quit();

        };

    shutdownGL();
}

void Animator::run(StepFunction&& step) {

    mRunning = true;

    using clock = high_resolution_clock;

    nanoseconds lag(0), elapsedTime(0);
    clock::time_point nowTime, oldTime = clock::now() - mLengthOfStep;
    std::uint32_t tickCount = 0;

    do {
        nowTime = clock::now();
        elapsedTime = nowTime - oldTime;
        oldTime = nowTime;

        lag += elapsedTime;
        while (lag >= mLengthOfStep) {

            step(mFloats);

            FloatSpaceConvert::floatSpaceConvert(mFloats, mPixels, mColorizeMode, 0.0f, 1.0f, *mSelectedStripes);

            render();

            lag -= mLengthOfStep;

            ++tickCount;
        }

        doEvents();

    } while (mRunning);
}

void Animator::animateStatic(std::size_t floatCount) {

    auto [width, height] = FloatSpaceConvert::getDimensions(floatCount, Animator::mAspectRatio);

    mTextureWidth = width;
    mTextureHeight = height;

    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);
    std::vector<float> floats(getSize());

    auto step = [&](auto floats) {

        std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
            return range(random);
            });
        };

    setup(floats);
    run(step);
}

std::size_t Animator::getSize() { return mTextureWidth * mTextureHeight; }

