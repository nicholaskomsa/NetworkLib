
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Animator.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    
    try {

        Animator animator;

      // animator.animateMT19937();
        animator.animateChatGPT2();
    }
    catch (const Animator::Error& e) {
        MessageBoxA(nullptr, e.what(), "Animator Error", MB_OK | MB_ICONERROR);
    }

    return 0;
}