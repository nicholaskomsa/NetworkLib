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
		//the most simple is just a parallel feature, (SectionFunctor)
		//next, there is a setup followed by a parallel feature: (SectionsFunctor, SectionFunctor)
		//next, there could be a setup, parallel, and a finale, (SectionsFunctor, SectionFunctor, SectionsFunctor)

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

		using Offsets = std::pair<std::size_t, std::size_t>;

		std::any mDefaultAny;

		struct Section {
			Offsets mOffsets;
			std::any mAny;

			Section(const std::any& any) : mAny(any) {};
		};

		using Sections = std::vector<Section>;
		using SectionsView = std::span<Section>;
		using SectionsFunctor = std::function<void(SectionsView)>;
		using SectionFunctor = std::function<void(Section&)>;

		Sections mSections;
		SectionsView mSectionsView;

		std::size_t mSize{ 0 };

		Parallel() = default;
		Parallel( std::size_t size, std::size_t hardwareSections = mHardwareThreads) {
			section(size, hardwareSections);
		}

		void setup(std::any defaultAny, std::size_t size=1, std::size_t hardwareSections = mHardwareThreads) {

			mDefaultAny = defaultAny;
			section(size, hardwareSections);
		}

		void section(std::size_t size=1, std::size_t hardwareSections = mHardwareThreads) {

			mSize = size;

			auto sectionSize = size / hardwareSections;
			if (sectionSize == 0) sectionSize = 1;
			auto numSections = size / sectionSize;

			std::size_t start = 0, end = 0;

			if (mSections.size() > numSections) {
				//shrink without destruct
			}
			else //grow
				mSections.resize(numSections, { mDefaultAny });

			mSectionsView = { mSections.begin(), numSections };

			for (std::size_t s = 0; s < numSections; ++s) {

				end = start + sectionSize;

				mSections[s].mOffsets = { start, end };

				start = end;
			}
			mSectionsView.back().mOffsets.second = size;
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
		void operator()(SectionsFunctor&& setup, SectionFunctor&& functor, bool single = false) {

			setup(mSectionsView);
			operator()(std::move(functor), single);
		}
		void operator()(SectionsFunctor&& setup, SectionFunctor&& functor, SectionsFunctor&& finale, bool single = false) {

			setup(mSectionsView);
			operator()(std::move(functor), single);
			finale(mSectionsView);
		}
	};
}