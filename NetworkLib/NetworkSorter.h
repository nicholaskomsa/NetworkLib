#pragma once

#include <map>

#include "CpuNetwork.h"

namespace NetworkLib {

	struct NetworksSorter {

		struct SortBySuperRadius {

			Cpu::NetworksMap* mNetworks = nullptr;

			bool operator()(const Cpu::Network& a, const Cpu::Network& b) const {
				return a.mMisses < b.mMisses || (a.mMisses == b.mMisses && a.mMse < b.mMse);
			}

			bool operator()(std::size_t a, std::size_t b) const {

				auto& networks = *mNetworks;
				        
				auto& nna = networks[a];
				auto& nnb = networks[b];

				return operator()(nna, nnb);
			}
		};

		using Id = std::size_t;
		using NetworksIds = std::vector<Id>;
		NetworksIds mNetworksIds;

		Cpu::NetworksMap* mNetworksMap;

		void create(Cpu::NetworksMap& networksMap){
			
			mNetworksMap = &networksMap;
			
			mNetworksIds.clear();
			mNetworksIds.reserve(networksMap.size());

			for (auto& [id, nn] : networksMap)
				mNetworksIds.push_back(id);
		}

		void destroy() {
			mNetworksIds.clear();
			mNetworksIds.shrink_to_fit();
			mNetworksMap = nullptr;
		}

		std::vector<std::size_t> getTop(std::size_t number=1) {
			std::vector<std::size_t> top;
			top.reserve(number);
			for (auto id : mNetworksIds | std::views::take(number))
				top.push_back(id);
			return top;
		}
		std::vector<std::size_t> getBottom(std::size_t number = 1) {
			std::vector<std::size_t> bottom;
			bottom.reserve(number);
			for (auto id : mNetworksIds | std::views::reverse | std::views::take(number))
				bottom.push_back(id);
			return bottom;
		} 
		Cpu::Network& getBest() {
			return (*mNetworksMap)[getBestId()];
		}
		Cpu::Network& getWorst() {
			return (*mNetworksMap)[getWorstId()];
		}
		std::size_t getBestId() {
			return mNetworksIds.front();
		}
		std::size_t getWorstId() {
			return mNetworksIds.back();
		}

		void sortBySuperRadius() {
			std::sort(mNetworksIds.begin(), mNetworksIds.end(), SortBySuperRadius{ mNetworksMap });
		}
	};
	class NetworksTracker {
	public:

		NetworksSorter mSorter;

		Cpu::NetworksMap mNetworksMap;
		std::size_t mTrackedMax = 1;

		void create(std::size_t trackedMax) {
			mTrackedMax = trackedMax;
		}
		void destroy() {
			mSorter.destroy();
			mNetworksMap.clear();
		}

		void track(const Cpu::Network& network) {
			
			auto exists = mNetworksMap.find(network.mId);

			if (exists != mNetworksMap.end()) {
				auto& current = exists->second;
				bool newWorst = NetworksSorter::SortBySuperRadius{}(network, current);
				if (newWorst)
					current = network;

			}
			else if(mNetworksMap.size() < mTrackedMax)
				mNetworksMap.insert({ network.mId, network });
			else {
				//a new network maybe better than the worst one
				auto& worst = mSorter.getWorst();
				bool newWorst = NetworksSorter::SortBySuperRadius{}(network, worst);
				if (newWorst)
					worst = network;
			}

			mSorter.create(mNetworksMap);
			mSorter.sortBySuperRadius();
		}

		void track(const Cpu::NetworksMap& networksMap) {

			for (auto& [id, network] : networksMap)
				track(network);
		}
	};
}
