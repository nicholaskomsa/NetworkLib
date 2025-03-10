#pragma once

#include <string>
#include <chrono>

using namespace std::chrono;
template<typename TimeType>
TimeType time(const std::string& caption, auto&& functor) {

	if (caption.size()) std::println("timing {}", caption);

	auto start = high_resolution_clock::now();
	functor();
	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<TimeType>(end - start);

	if (caption.size()) std::println("\t{} took {}", caption, elapsed.count());

	return elapsed;
}

template<typename TimeType>
struct TimeAverage {

	TimeType sum = TimeType(0);
	std::size_t count = 0;

	void accumulateTime(auto&& functor) {
		sum += time<TimeType>("", std::move(functor));
		++count;
	}
	std::size_t average() const {
		return sum.count() / count;
	}
};
