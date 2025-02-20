#include <iostream>

#include <Gpt2.h>
#include <FloatSpaceConvert.h>

int main() {

	using GPT2 = NetworkLib::GPT2;

	try {
		GPT2 gpt2;
		
		gpt2.setup();

		gpt2.chat();
	}
	catch (const GPT2::Error& e) {
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