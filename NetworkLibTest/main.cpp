#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

#include <CpuNetwork.h>
#include <CpuTensor.h>
#include <ModelLogic.h>
#include <ModelConv1.h>
#include <ModelMNIST.h>

int main() {

	//using GPT2 = NetworkLib::GPT2;

//	GPT2::Diagnostics diag;
	//diag.feedForwardSpeed1024();

	//diag.feedForwardSpeed1024();
	//diag.backwardTest64();

	//diag.SGDTest();
	//diag.serializeTest();
	//diag.simpleChat();

	//NetworkLib::Cpu::example();
	//NetworkLib::Model::XOR xorModel;
	//xorModel.run();
	//NetworkLib::Model::XORLottery xorLotteryModel;
	//xorLotteryModel.run();
	//NetworkLib::Model::LogicLottery logicLotteryModel;
	//logicLotteryModel.run();

	//NetworkLib::Gpu::Environment::example();
	//NetworkLib::Gpu::Environment::example2();

//	NetworkLib::Model::XOR xorModel;
//	xorModel.run();
	//NetworkLib::Model::LogicLottery logicLotto;
	//logicLotto.run();
	//NetworkLib::Model::Convolution1Comparison comparison;
	//comparison.run();
	//NetworkLib::Model::Convolution1Lottery conv1Lotto;
	//conv1Lotto.run();

	NetworkLib::Model::MNIST mnistModel;
	mnistModel.run();
	//NetworkLib::Model::MNISTLottery mnistLottery;
	//mnistLottery.run();

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}