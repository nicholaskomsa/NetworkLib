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

	NetworkLib::FloatSpace1 floatSpace1;
	floatSpace1.resize(5 + 5 * 5);

	auto begin = floatSpace1.mFloats.begin();
	NetworkLib::View1 v1;
	NetworkLib::View2 v2;

	v1.advance(begin, 5);
	v2.advance(begin, 5, 5);

	v2[2, 3] = 6;

	std::cout << v2[2, 3];
	std::cout << v1[2];

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}