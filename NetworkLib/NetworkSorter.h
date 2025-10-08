#pragma once


#include "CpuNetwork.h"

namespace NetworkLib {

	struct NetworksSorter {

		using Idx = std::size_t;
		using NetworksIdx = std::vector<Idx>;
		NetworksIdx mNetworksIdx;

		Cpu::NetworksView mNetworks;

		void create(Cpu::NetworksView networks) {
			mNetworks = networks;
			mNetworksIdx.resize(mNetworks.size());
			std::iota(mNetworksIdx.begin(), mNetworksIdx.end(), 0);
		}

		void destroy() {
			mNetworks = {};
			mNetworksIdx.clear();
			mNetworksIdx.shrink_to_fit();
		}

		Cpu::Network& getBest() {
			return mNetworks[getBestIdx()];
		}
		std::size_t getBestIdx() {
			return mNetworksIdx.front();
		}
		void sortByMse() {
			std::sort(mNetworksIdx.begin(), mNetworksIdx.end(), [&](auto& a, auto& b) {
				return mNetworks[a].mMse < mNetworks[b].mMse;
				});
		}
		void sortBySuperRadius() {
			std::sort(mNetworksIdx.begin(), mNetworksIdx.end(), [&](auto& a, auto& b) {

				auto& nna = mNetworks[a];
				auto& nnb = mNetworks[b];

				return nna.mMisses < nnb.mMisses || (nna.mMisses == nnb.mMisses && nna.mMse < nnb.mMse);
				});
		}
	};

}
