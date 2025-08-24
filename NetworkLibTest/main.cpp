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
	std::size_t a = 5, b = 5, c = 5;
	floatSpace1.resize(a + b*c + b*c);

	auto begin = floatSpace1.mFloats.begin();
	NetworkLib::View1 v1;
	NetworkLib::View2 v2;

	using FixedView2 = std::mdspan<NetworkLib::FloatType, std::extents<NetworkLib::Dimension, 5ULL, 5ULL>>;

	FixedView2 fv2(nullptr);

	NetworkLib::dynamicAdvance(v1, begin, a);
	NetworkLib::dynamicAdvance(v2, begin, b, c);
	NetworkLib::fixedAdvance(fv2, begin, b, c);

	v2[b-1, c-1] = 6;
	floatSpace1.mView[2] = 7;
	fv2[0, 2] = 9;

	std::println("{} {} {} {}", v2[b - 1, c - 1], v1[2], floatSpace1.mView[30-1], fv2[0,2]);

	auto v2Shape = NetworkLib::getShape(v2);
	for (auto d : v2Shape)
		std::print("{} ", d);

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}