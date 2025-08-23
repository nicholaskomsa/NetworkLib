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
	
	NetworkLib::Tensor1 t1;
	t1.create(5);
	t1.at(1) = 7;
//
	std::cout << t1.at(1);

	NetworkLib::Floats floats(10);
	auto begin = floats.begin();
	NetworkLib::TensorView1 tv1, tv2;
	tv1.advance(begin, 5);
	tv2.advance(begin, 5);

	tv1.at(1) = 5;
	tv2.at(1) = 10;

	std::cout << tv1.at(1);

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}