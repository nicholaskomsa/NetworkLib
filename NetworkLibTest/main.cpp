#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

#include <CpuNetwork.h>
#include <CpuTensor.h>
#include <Model.h>

#include <CpuNetwork.h>
#include <GpuNetwork.h>

int main() {

	using GPT2 = NetworkLib::GPT2;

	GPT2::Diagnostics diag;
	//diag.feedForwardSpeed1024();

	//diag.feedForwardSpeed1024();
	//diag.backwardTest64();

	//diag.SGDTest();
	//diag.serializeTest();
	//diag.simpleChat();

	//NetworkLib::Cpu::example();
	//NetworkLib::Cpu::networkExample();
	//NetworkLib::Model::XOR xorModel;
	//xorModel.run();
	NetworkLib::Model::XORLottery xorLotteryModel;
	xorLotteryModel.run();

	//NetworkLib::Gpu::Environment::example();

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}