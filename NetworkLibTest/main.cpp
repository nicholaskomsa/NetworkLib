#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	GPT2 gpt2;
	gpt2.readSafeTensors();
	FloatSpaceConvert::colorizeFloatSpace("gpt2", gpt2.mFloatSpace);

	std::puts("Program Finished press enter to exit");
	std::cin.get();

	return 0;
}