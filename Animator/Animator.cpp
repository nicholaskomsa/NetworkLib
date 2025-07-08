#include "Animator.h"

#include <format>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <execution>

#include <Gpt2.h>
#include <Serializer.h>

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

void Animator::render() {

    glBindTexture(GL_TEXTURE_2D, mTexture);
    const auto& [coord,dims] = mFloatSubSpaceDimensions;
    auto& [width, height] = dims;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mPixels.data());
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

    auto quit = [&]() {
        return type == SDL_EVENT_QUIT
            || type == SDL_EVENT_KEY_DOWN && key == SDLK_ESCAPE;
        };
    auto keydown = [&]() {
        return type == SDL_EVENT_KEY_DOWN;
        };

    while (SDL_PollEvent(&event)) {

        if (quit()) {

            mRunning = false;
            return;
        
        }else if(keydown()) {

            keyRepeatGuard([&]() {

                bool doChangeDimensions = false;
                bool doConvert = false;
                float translateSpeed = mTranslateSpeed / mScale;

                switch (key) {
                case SDLK_1: 
                    doConvert = true;  mColorizeMode = ColorizeMode::NICKRGB; break;
                case SDLK_2: 
                    doConvert = true; mColorizeMode = ColorizeMode::SHORTNRGB; break;
                case SDLK_3: 
                    doConvert = true; mColorizeMode = ColorizeMode::ROYGBIV; break;
                case SDLK_4: 
                    doConvert = true; mColorizeMode = ColorizeMode::GREYSCALE; break;
                case SDLK_5: 
                    doConvert = true; mColorizeMode = ColorizeMode::BINARY; break;

                case SDLK_Q:
                    doConvert = true;
                    if (mSelectedStripes == mStripes.begin())
                        mSelectedStripes = std::prev(mStripes.end(), 1);
                    else
                        mSelectedStripes = std::prev(mSelectedStripes, 1);
                    break;
                case SDLK_W:
                    doConvert = true;
                    std::advance(mSelectedStripes, 1);
                    if (mSelectedStripes == mStripes.end())
                        mSelectedStripes = mStripes.begin();
                    break;

                case SDLK_LEFT:
                    doConvert = true;
                    doChangeDimensions = true;
                    mX -= translateSpeed;
                    break;
                case SDLK_RIGHT:
                    doConvert = true;
                    doChangeDimensions = true;
                    mX += translateSpeed;
                    break;
                case SDLK_UP:
                    doConvert = true;
                    doChangeDimensions = true;
                    mY += translateSpeed;
                    break;
                case SDLK_DOWN:
                    doConvert = true;
                    doChangeDimensions = true;
                    mY -= translateSpeed;
                    break;
                case SDLK_A:
                    doConvert = true;
                    doChangeDimensions = true;
                    mScale /= 2.0f;
                    break;
                case SDLK_S:
                    doChangeDimensions = true;
                    doConvert = true;
                    mScale *= 2.0f;
                    break;
                case SDLK_R:
                    doChangeDimensions = true;
                    doConvert = true;
                    mX = 0.0f;
                    mY = 0.0f;
                    mScale = 1.0f;
                    break;

                case SDLK_SPACE:
                    mPaused = !mPaused;
                    break;
                }

                if (doChangeDimensions)
                    setDimensions();
                
                if (doConvert)
                    floatSpaceConvert();
                
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
void Animator::updateCamera() {

    auto setOrthoProjection = [&]() {
                
        float left = -1, right = 1
            , top = 1, bottom = -1
            , n = -1, f = 1;

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
}
void Animator::setup(FloatsView floats = {}) {

    mFloats = floats;

    //mPixels.resize(mTextureWidth * mTextureHeight, 0.0f);

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

            updateCamera();
            };

            auto createQuad = [&]() {

                std::array<float, 16> vertices = {
                    //vertex = X, Y, U, V
                    -1.0f, 1.0f,    0.0f, 0.0f  // Top-left
                    , 1.0f, 1.0f,     1.0f, 0.0f  // Top-right
                    , -1.0f, -1.0f,   0.0f, 1.0f  // Bottom-left
                    , 1.0f, -1.0f,    1.0f, 1.0f };   // Bottom-right

                glGenVertexArrays(1, &mVao);
                glGenBuffers(1, &mVbo);

                glBindVertexArray(mVao);
                glBindBuffer(GL_ARRAY_BUFFER, mVbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

                // Position Attribute
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
                glEnableVertexAttribArray(0);

                // Texture Coordinate Attribute
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
                glEnableVertexAttribArray(1);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindVertexArray(0);
                };

        setupModernGL();
        createQuad();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        };

    initGL();

    mSelectedStripes = mStripes.begin();

    setDimensions();
    floatSpaceConvert();
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

            if (!mPaused) {

                if( step(mFloats) )
                    floatSpaceConvert();
            }

            render();

            lag -= mLengthOfStep;

            ++tickCount;
        }

        doEvents();

    } while (mRunning);
}
std::size_t Animator::getSize() { return mTextureWidth * mTextureHeight; }


void Animator::animateMT19937(std::size_t floatCount) {

    auto [width, height] = FloatSpaceConvert::getDimensions(floatCount, Animator::mAspectRatio);

    mTextureWidth = width;
    mTextureHeight = height;

    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);
    std::vector<float> floats(getSize());


   NetworkLib::Serializer serializer;
   serializer.createOutputStream(floats, mFloatSubSpaceDimensions, mTextureWidth);

    auto step = [&](auto floats) {

        std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
            return range(random);
            });

        serializer.writeToFile();
        return true;
        };

    setup(floats);
    run(step);
}

void Animator::viewChatGPT2() {

    auto gpt2 = std::make_unique<NetworkLib::GPT2>(); //gpt2 is large and offsourced to heap

    gpt2->setup();

    auto tensorSpace = gpt2->getForward().getTensorSpace();

    auto [width, height] = FloatSpaceConvert::getDimensions(tensorSpace.size(), Animator::mAspectRatio);

    mTextureWidth = width;
    mTextureHeight = height;
    mPaused = true;

    auto step = [&](auto floats) {
        return false;
        };

    setup(tensorSpace);
    run(step);
}

void Animator::animateChatGPT2() {

    NetworkLib::Serializer serializer;

    serializer.createInputStream();

    auto [width, height] = FloatSpaceConvert::getDimensions(serializer.getStreamFrameSize(), Animator::mAspectRatio);

    mTextureWidth = width;
    mTextureHeight = height;
    mPaused = true;

    FloatSpaceConvert::FloatSpaceDimensions oldDimensions;

    auto step = [&](auto floats) {

        if( oldDimensions != mFloatSubSpaceDimensions ) {

            oldDimensions = mFloatSubSpaceDimensions;

            serializer.mFrameRect = mFloatSubSpaceDimensions;
        }

        auto frame = serializer.getCurrentFrame();
        if (frame.has_value()) {
            mFloats = frame.value();
            return true;
        }

        return false;
        };

    setup();

    serializer.startReadingWindow(mFloatSubSpaceDimensions, mTextureWidth );

    run(step);
}