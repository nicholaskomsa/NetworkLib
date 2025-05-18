#pragma once

#include <string>
#include <chrono>

namespace NetworkLib {

	using namespace std::chrono;


	template<typename TimeType>
	TimeType time(auto&& functor) {

		auto start = high_resolution_clock::now();
		functor();
		auto end = high_resolution_clock::now();
		auto elapsed = duration_cast<TimeType>(end - start);

		return elapsed;
	}

	template<typename TimeType>
	TimeType time(const std::string& caption, auto&& functor) {

		std::println("timing {}", caption);

		auto elapsed = time<TimeType>(std::move(functor));

		std::println("\t{} took {}", caption, elapsed.count());

		return elapsed;
	}

	template<typename TimeType>
	struct TimeAverage {

		TimeType sum = TimeType(0);
		std::size_t count = 0;

		void accumulateTime( TimeType time) {
			sum += time;
			++count;
		}

		TimeType accumulateTime(auto&& functor) {
			
			auto elapsed = time<TimeType>(std::move(functor));
			accumulateTime(elapsed);
			return elapsed;
		}


		std::size_t average() const {
			return sum.count() / count;
		}
		void reset() {
			sum = TimeType(0);
			count = 0;
		}

		std::string getString() {

			const auto str = std::format("{}x{}={}", average(), count, sum );
			reset();

			return str;
			};
	};
}
