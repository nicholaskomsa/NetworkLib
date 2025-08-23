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
	
	NetworkLib::FloatSpace1 t1;
	t1.resize(5);


	NetworkLib::Floats floats(5 + 5*5);
	auto begin = floats.begin();
	NetworkLib::View1D tv1; 
	NetworkLib::View2D tv2;

	NetworkLib::advance(tv1, begin, 5);
	
	NetworkLib::at(tv1, 2);



	//tv1.advance(begin, 5);
	//tv2.advance(begin, 5, 5);


	//std::cout << tv2.at(1,4);

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}