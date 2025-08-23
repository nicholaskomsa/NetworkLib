#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

#include <CpuNetwork.h>

int main() {

	using GPT2 = NetworkLib::GPT2;

	GPT2::Diagnostics diag;
	//diag.feedForwardSpeed1024();

	//diag.feedForwardSpeed1024();
	//diag.backwardTest64();

	//diag.SGDTest();
	//diag.serializeTest();
	//diag.simpleChat();
	
	NetworkLib::Tensor2 t2;
	t2.create(5,5);
	t2.at(1,1) = 7;
//
	std::cout << t2.at(1,1);

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}