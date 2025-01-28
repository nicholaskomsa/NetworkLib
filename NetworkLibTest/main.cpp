#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	GPT2 gpt2;
	
	try {
		gpt2.readSafeTensors();
		//FloatSpaceConvert::colorizeFloatSpace("gpt2", gpt2.mFloatSpace);
		gpt2.mDecoder.readEnc();
		gpt2.mData.readData();
		//auto text = gpt2.mDecoder.decode(gpt2.mData.mTokens);
	//	std::cout << text;

		//for( auto word : gpt2.mDecoder.mWords )
		//	std::cout << word << std::endl;

		gpt2.feedForward();

	}catch(const GPT2::Error& e){
		std::println(std::cerr, "{}", e.what());
	}catch(...){
		std::println(std::cerr, "Unknown error");
	}

	std::puts("Program Finished press enter to exit");
	std::cin.get();

	return 0;
}