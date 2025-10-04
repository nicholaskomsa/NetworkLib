#include "Animator.h"

#include <format>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <execution>

#include <Gpt2.h>
#include <Serializer.h>
#include <Model.h>

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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(mQuadManager.mVao);

    auto renderViewer = [&]() {

        glBindTexture(GL_TEXTURE_2D, mViewerTexture);
        const auto& [coord, dims] = mFloatRect;
        auto& [width, height] = dims;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mPixels.data());

        mQuadManager.render(mViewerQuadRef);
        };

    renderViewer();

    mTextManager.render();

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

    auto keydown = [&]() {
        return type == SDL_EVENT_KEY_DOWN;
        };

    auto quit = [&]() {
        return type == SDL_EVENT_QUIT
            || keydown() && key == SDLK_ESCAPE;
        };

    while (SDL_PollEvent(&event)) {

        if (quit()) {

            mRunning = false;
            return;
        
        }else if(keydown()) {

            keyRepeatGuard([&]() {

                bool doChangeDimensions = true;
                bool doConvert = true;
                float translateSpeed = mTranslateSpeed / mScale;

                switch (key) {
                case SDLK_LEFT:
                    mX -= translateSpeed;
                    break;
                case SDLK_RIGHT:
                    mX += translateSpeed;
                    break;
                case SDLK_UP:
                    mY += translateSpeed;
                    break;
                case SDLK_DOWN:
                    mY -= translateSpeed;
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
                default:
                    doChangeDimensions = false;
                }

                switch (key) {
                case SDLK_1: 
                    mColorizeMode = ColorizeMode::NICKRGB; break;
                case SDLK_2: 
                    mColorizeMode = ColorizeMode::SHORTNRGB; break;
                case SDLK_3: 
                    mColorizeMode = ColorizeMode::ROYGBIV; break;
                case SDLK_4: 
                    mColorizeMode = ColorizeMode::GREYSCALE; break;
                case SDLK_5: 
                    mColorizeMode = ColorizeMode::BINARY; break;

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
                default:
                    doConvert = false;
				}

                switch(key){
                case SDLK_SPACE:
                    mPaused = !mPaused;
                    break;
                case SDLK_BACKSPACE:
                    mTextManager.toggleVisibility();
                    break;
                }

                if (mCustomGuiEvents)
                    mCustomGuiEvents(doChangeDimensions, doConvert);

                if (doChangeDimensions )
                    setDimensions();
                
                if (doConvert)
                    floatSpaceConvert();
                
                });
        }
    }
}
Animator::Animator(std::size_t width, std::size_t height) {
    mFrameWidth = width;
    mFrameHeight = height;
}
Animator::~Animator() {
    shutdown();
}

void Animator::setup(FloatsView floats = {}) {

    mFloats = floats;
    mSelectedStripes = mStripes.begin();

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
            };

        auto setupQuads= [&](){

            mViewerQuadRef = mQuadManager.addIdentity();

            mTextManager.create(&mQuadManager, mTextScale);
            auto& minecraftFont = mTextManager.mMinscraftFontFace;

            mTextAreaRef = mTextManager.addTextArea();
            auto& textArea = mTextManager.getTextArea(mTextAreaRef);
            
            mTicksValueRef = textArea.addLabeledValue(minecraftFont, "Ticks:", "0");
            mColorModeValueRef = textArea.addLabeledValue(minecraftFont, "Color Mode:", 
                std::format("{}x{}"
                , FloatSpaceConvert::getColorNames()[mColorizeMode]
                , *mSelectedStripes));

            if (mCreateCustomGui)
                mCreateCustomGui();

            textArea.create(mQuadManager, mTextScale);
            mQuadManager.generate(); 

            textArea.render(mQuadManager, true);
        };

        setupModernGL();
        setupQuads();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        };

    initGL();

    setDimensions();
    floatSpaceConvert();
}
void Animator::shutdown() {

    auto shutdownGL = [&]() {

        glDeleteTextures(1, &mViewerTexture);

        mTextManager.destroy();
        mQuadManager.destroy();

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

    auto& textArea = mTextManager.getTextArea(mTextAreaRef);

    do {
        nowTime = clock::now();
        elapsedTime = nowTime - oldTime;
        oldTime = nowTime;

        lag += elapsedTime;
        while (lag >= mLengthOfStep) {

            textArea.updateLabeledValue(mTicksValueRef, std::to_string(tickCount));
            if (mCustomGuiRender) mCustomGuiRender();

            if (!mPaused && step(mFloats) )
                floatSpaceConvert();

            render();

            lag -= mLengthOfStep;

            ++tickCount;
        }

        doEvents();

    } while (mRunning);
}
std::size_t Animator::getSize() { return mFrameWidth * mFrameHeight; }


void Animator::animateMT19937(std::size_t floatCount) {

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(floatCount, Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    std::mt19937 random;
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);
    std::vector<float> floats(getSize());

    setup(floats);

    auto step = [&](auto floats) {

        std::generate(std::execution::seq, floats.begin(), floats.end(), [&]() {
            return range(random);
            });

        return true;
        };

    run(step);
}

void Animator::viewChatGPT2() {

    auto gpt2 = std::make_unique<NetworkLib::GPT2>(); //gpt2 is large and offsourced to heap

    gpt2->setup();

    auto tensorSpace = gpt2->getForward().getTensorSpace();

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(tensorSpace.size(), Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;
   
    mPaused = true;

    setup(tensorSpace);

    auto step = [&](auto floats) {
        return false;
        };

    run(step);
}

void Animator::animateChatGPT2() {

    NetworkLib::Serializer serializer;

    auto [frameWidth, frameHeight] = serializer.createInputStream();
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    mScale = 4.0f;
    setup();

    serializer.readBackBuffer(mFloatRect);

    auto step = [&](auto floats) {

        auto frame = serializer.getCurrentFrame(mFloatRect);
        if (frame.has_value()) {
            mFloats = frame.value();
            return true;
        }
        else
            mPaused = true;

        return false;
        };


    run(step);
}
void Animator::animateXORNetwork() {

    NetworkLib::Model::XOR xorModel;
    xorModel.create();
    auto& [gpu, gpuNetwork] = *xorModel.mGpuTask;

    std::size_t generation = 0;
    
    auto selectedView = gpuNetwork.mWeights;
    auto selectedFloatsView = NetworkLib::Cpu::Tensor::view(selectedView.mView);

    auto [frameWidth, frameHeight] = FloatSpaceConvert::getDimensions(selectedFloatsView.size(), Animator::mAspectRatio);
    mFrameWidth = frameWidth;
    mFrameHeight = frameHeight;

    TextArea::LabeledValueReference mMseValueRef, mAccuracyValueRef;

    mCreateCustomGui = [&]() {

        auto& textArea = mTextManager.getTextArea(mTextAreaRef);

        mMseValueRef = textArea.addLabeledValue(mTextManager.mMinscraftFontFace, "Mse:", "    ");
        mAccuracyValueRef = textArea.addLabeledValue(mTextManager.mMinscraftFontFace, "Accuracy:", "000" );
        };

    mCustomGuiEvents = [&](bool& changeDimensions, bool& doConvert){

        };

    float mse = 0.0f;
    std::size_t misses = 0;

    mCustomGuiRender = [&]() {
        auto& textArea = mTextManager.getTextArea(mTextAreaRef);
        textArea.updateLabeledValue(mMseValueRef, std::to_string(mse));

       auto sampleNum = xorModel.mBatchedSamplesView.size() * xorModel.mBatchSize;
        float accuracy = (sampleNum - misses) / float(sampleNum) * 100.0f;
        textArea.updateLabeledValue(mAccuracyValueRef, std::format("{}", accuracy));
        };

    setup(selectedFloatsView);

    auto step = [&](auto floats) {

        xorModel.train();

        xorModel.calculateConvergence();
        gpu.downloadConvergenceResults();
        mse = gpu.getMseResult();
		misses = gpu.getMissesResult();

        return true;
        };

    run(step);

    xorModel.destroy();
}