#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	using GPT2 = NetworkLib::GPT2;
	GPT2 gpt2;
	
	try {

		gpt2.readSafeTensors();
		//FloatSpaceConvert::colorizeFloatSpace("gpt2", gpt2.mFloatSpace);
		
		auto& decoder = gpt2.mDecoder;

		decoder.readEnc();
		
		gpt2.mData.readData();

		auto& tokens = gpt2.mData.mTokens;

		//GPT2::TokensView tokensView(tokens.end() - GPT2::mTestInputSize, tokens.end());
		GPT2::TokensView tokensView(tokens.begin()+ GPT2::mTestInputSize, GPT2::mTestInputSize);

		gpt2.slide(tokensView, 20000);

	}catch(const NetworkLib::GPT2::Error& e){
		std::println(std::cerr, "{}", e.what());
	}catch(...){
		std::println(std::cerr, "Unknown error");
	}

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}