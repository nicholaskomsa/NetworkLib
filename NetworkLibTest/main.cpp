#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	using GPT2 = NetworkLib::GPT2;

	GPT2::Diagnostics diag;
	//diag.firstCitizenTest64();

	//diag.feedForwardSpeed1024();
	//diag.backwardTest64();

	//diag.SGDTest();
	diag.serializeTest();
	//diag.simpleChat();

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}