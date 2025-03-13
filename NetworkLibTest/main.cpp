#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	using GPT2 = NetworkLib::GPT2;

	GPT2::Diagnostics diag;
	diag.feedForwardSpeed1024();

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}