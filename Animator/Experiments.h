#pragma once

#include <Animator.h>

class AnimateMT19937 : public Animator {
public:
	AnimateMT19937() = default;
	void run();
};

class AnimatorViewChatGPT2 : public Animator {
public:
    AnimatorViewChatGPT2() = default;
	void run();
};

class AnimatorChatGPT2 : public Animator {
public:
	AnimatorChatGPT2() = default;
	void run();
};

class AnimatorXOR : public Animator {
public:
	AnimatorXOR() = default;
	void run();
};

class AnimatorMNIST : public Animator {
public:
	AnimatorMNIST() = default;
	void run();
};
