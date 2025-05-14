#pragma once

#include <span>
#include <functional>
#include <any>
#include <algorithm>
#include <execution>

namespace NetworkLib {

	struct Parallel {

		//this Parallal object is a functor which executes a parallel for loop over
		//each parallel "section". Each section aims to complete a for loop "a to b" 
		//for a total of N iterations
		
		//there are three types of operations which enable different amounts of 
		//parallel-customization.

		//there is an any object which represents the specific parallel data and is left entirely to external use
		
		//the section function is used to setup the specific parallel "a to b" sections and must be called before functor
		//you pass your total data size "N" to section, and it will be iterated over, in parallel, in specific "a to b" sections.
		//the "a to b" interface follows a simple process:
		//Parallel parallel(N);
		//parallel([&](auto& section){
		// 
		//	auto& [ a, b ] = section.mOffsets;
		//
		//	for(auto i = a; i < b; ++i){
		//		func(i);
		//	}
		// });
		
		//when section is called it is passed a requested number of hardware sections, there will be at least this many sections 

		static constexpr auto mHardwareThreads = 8, mLargeHardwareThreads = 32;


		std::any mDefaultAny;

		struct Section {

			using IotaView = std::ranges::iota_view<std::size_t, std::size_t>;
			IotaView mIotaView;

			std::any mAny;

			Section(const std::any& any) : mAny(any) {};
		};

		using Sections = std::vector<Section>;
		using SectionsView = std::span<Section>;
		using SectionsFunctor = std::function<void(SectionsView)>;
		using SectionFunctor = std::function<void(Section&)>;

		Sections mSections;
		SectionsView mSectionsView;

		std::size_t mSize{ 0 }, mSectionNum{ 1 };

		Parallel() = default;
		Parallel( std::size_t size, std::size_t sectionNum = mHardwareThreads) {

			mSectionNum = sectionNum;
			section(size);
		}

		void setup(std::any defaultAny, std::size_t size=1, std::size_t sectionNum = mHardwareThreads) {

			mDefaultAny = defaultAny;
			mSectionNum = sectionNum;

			section(size);
		}

		void section(std::size_t size) {

			mSize = size;

			auto sectionSize = size / mSectionNum;
			auto sectionNum = mSectionNum;

			if (sectionSize == 0) {
				sectionNum = size;
				sectionSize = 1;
			}

			std::size_t start = 0, end = 0;

			if (mSections.size() > sectionNum) {
				//shrink without destruct
			}
			else //grow
				mSections.resize(sectionNum, { mDefaultAny });

			mSectionsView = { mSections.begin(), sectionNum };

			for (std::size_t s = 0; s < sectionNum - 1; ++s) {

				end = start + sectionSize;

				mSections[s].mIotaView = std::ranges::iota_view(start, end);

				start = end;
			}
			mSections.back().mIotaView = std::ranges::iota_view(start, size);
		}

		void operator()(SectionFunctor&& functor, bool single = false) {

			if (single)
				std::for_each(std::execution::seq, mSectionsView.begin(), mSectionsView.end(), [&](auto& section) {
					functor(section);
					});
			else
				std::for_each(std::execution::par_unseq, mSectionsView.begin(), mSectionsView.end(), [&](auto& section) {
					functor(section);
					});
		}

		//setup done insde functor(parallel), followed by single threaded finale
		void operator()(SectionFunctor&& functor, SectionFunctor&& finale, bool single = false) {
			operator()(std::move(functor), single);
			operator()(std::move(finale), true);
		}

		//single threaded setup, parallel, single threaded finale
		void operator()(SectionFunctor&& setup, SectionFunctor&& functor, SectionFunctor&& finale, bool single = false) {

			operator()(std::move(setup), true);
			operator()(std::move(functor), single);
			operator()(std::move(finale), true);
		}


	};
}