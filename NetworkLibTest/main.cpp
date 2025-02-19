#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	using GPT2 = NetworkLib::GPT2;
	GPT2 gpt2;
	
	try {

		gpt2.readSafeTensors();
		//FloatSpaceConvert::colorizeFloatSpace("gpt2", gpt2.mFloatSpace);

		gpt2.mDecoder.readEnc();
		
	//	gpt2.mData.readData();

	//	auto& tokens = gpt2.mData.mTokens;

	//	GPT2::TokensView tokensView(tokens.begin(), GPT2::mTestInputSize * 0.9);

	//	std::print("{}",gpt2.mDecoder.decode(tokensView));
		
	//	gpt2.slide({ tokensView.begin(), tokensView.end() });

		gpt2.chat();
	}
	catch (const NetworkLib::GPT2::Error& e) {
		std::println(std::cerr, "{}", e.what());
	}catch( const std::exception& e){
		std::println(std::cerr, "{}", e.what());
	}catch(...){
		std::println(std::cerr, "Unknown error");
	}

	std::puts("\nProgram Finished press enter to exit");
	std::cin.get();

	return 0;
}