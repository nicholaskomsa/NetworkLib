
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Animator.h"

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