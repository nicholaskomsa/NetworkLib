#pragma once

#include <span>
#include <functional>
#include <any>
#include <algorithm>
#include <execution>

namespace NetworkLib {

	struct Parallel {

		//static constexpr auto mLargeHardwareThreads = 64;
		static constexpr auto mHardwareThreads = 8, mLargeHardwareThreads = 32;

		using Offsets = std::pair<std::size_t, std::size_t>;

		struct Section {
			Offsets mOffsets;
			std::any mAny;
		};

		using Sections = std::vector<Section>;
		using SectionsView = std::span<Section>;
		using SectionsFunctor = std::function<void(SectionsView)>;
		using SectionFunctor = std::function<void(Section&)>;

		Sections mSections;
		SectionsView mSectionsView;

		std::size_t mSize{ 0 };

		Parallel() = default;
		Parallel(std::size_t size, std::size_t hardwareSections = 0) {

			section(size, hardwareSections);
		};

		void section(std::size_t size, std::size_t hardwareSections = 0) {

			hardwareSections = hardwareSections == 0 ? mHardwareThreads : hardwareSections;

			mSize = size;

			auto sectionSize = size / hardwareSections;
			if (sectionSize == 0) sectionSize = 1;
			auto numSections = size / sectionSize;

			std::size_t start = 0, end = 0;

			if (mSections.size() > numSections) {
				//shrink without destruct
			}
			else //grow
				mSections.resize(numSections, { {}, {} });

			mSectionsView = { mSections.begin(), numSections };

			std::size_t s = 0;
			for (s = 0; s < numSections; ++s) {

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
		void operator()(SectionsFunctor&& sectionsFunctor, SectionFunctor&& functor, bool single = false) {

			sectionsFunctor(mSectionsView);
			operator()(std::move(functor), single);
		}
	};
}