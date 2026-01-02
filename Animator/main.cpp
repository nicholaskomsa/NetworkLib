
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <Animator.h>
#include "AnimateEquation.h"

#include "Experiments.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    
    try {

      //  AnimateMT19937 mt;
      //  mt.run();
      // 
        AnimatorXOR axor;
		axor.run();
       // animator.animateChatGPT2();
      //  animator.animateXORNetwork();
      //  animator.animateMNISTNetwork();
        
     //  EquationAnimator equationAnimator;
      // equationAnimator.run();
    }
    catch (const Animator::Error& e) {
        MessageBoxA(nullptr, e.what(), "Animator Error", MB_OK | MB_ICONERROR);
    }

    return 0;
}