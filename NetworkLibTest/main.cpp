#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

#include <CpuNetwork.h>
#include <CpuTensor.h>


int main() {

	using GPT2 = NetworkLib::GPT2;

	GPT2::Diagnostics diag;
	//diag.feedForwardSpeed1024();

	//diag.feedForwardSpeed1024();
	//diag.backwardTest64();

	//diag.SGDTest();
	//diag.serializeTest();
	//diag.simpleChat();

	namespace CpuTensor = NetworkLib::CpuTensor;
	CpuTensor::example();

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}