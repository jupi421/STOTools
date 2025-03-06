#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <ranges>
#include <numeric>
#include <Eigen/Core>
#include <iostream>

#define print(x) std::cout << x << std::endl

using Positions = std::vector<Eigen::Vector3d>;
using Position = Eigen::Vector3d;
using NNIDs = std::vector<size_t>;

namespace PolFinder {
	struct AtomPositions {
		Positions SrPositions {};
		Positions TiPositions {};
		Positions OPositions {};

		AtomPositions() {};
		AtomPositions(const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
			SrPositions.reserve(N_Sr);
			TiPositions.reserve(N_Ti);
			OPositions.reserve(N_O);
		}
	};

	struct NearestNeighbors {
		std::vector<std::vector<size_t>> Sr_NN_ids;
		std::vector<std::vector<size_t>> Ti_NN_ids;
		std::vector<std::vector<size_t>> O_NN_ids;
	};

	namespace helper {
		inline auto find_n_nearest = [](const Position &atom1, const Positions &atom_arr, const size_t n) {
			std::vector<double> diff { };
			diff.reserve(atom_arr.size());
			
			std::ranges::for_each(atom_arr.begin(), atom_arr.end(), [&atom1, &diff](const Position &atom2) {
				diff.push_back((atom2 - atom1).norm());
			});
			
			std::vector<size_t> atom_arr_ids(atom_arr.size());
			std::iota(atom_arr_ids.begin(), atom_arr_ids.end(), 0);
			
			std::ranges::sort(atom_arr_ids.begin(), atom_arr_ids.end(), [&diff](const size_t idx1, const size_t idx2) {
				return diff.at(idx1) < diff.at(idx2);
			});

			NNIDs n_min_instances;
			n_min_instances.reserve(n);
			std::ranges::copy_n(atom_arr_ids.begin(), n, std::back_inserter(n_min_instances));

			return n_min_instances;
		};

		inline auto get_NN = [](const AtomPositions &atom_arr, const size_t n, NearestNeighbors &NN_arr, const Position &ref_atom){
			NNIDs ref_atom_Sr_ids { find_n_nearest(ref_atom, atom_arr.SrPositions, n) };
			NNIDs ref_atom_Ti_ids { find_n_nearest(ref_atom, atom_arr.TiPositions, n) };
			NNIDs ref_atom_O_ids { find_n_nearest(ref_atom, atom_arr.OPositions, n) };

			NN_arr.Sr_NN_ids.push_back(ref_atom_Sr_ids);
			NN_arr.Ti_NN_ids.push_back(ref_atom_Ti_ids);
			NN_arr.O_NN_ids.push_back(ref_atom_O_ids);
		};
	}
	

	// write different behavior for other filetypes (CONTCAR, XDATCAR, xyz, ...), calculate atom numbers??? for end file and then parallelised for loop 
	inline Positions loadPosFromFile(std::string filename, uint head, const char* filetype = "POSCAR") {
		std::ifstream file { filename };
		
		if (!file.is_open()) {
			throw std::runtime_error("Failed loading file!");
		}

		std::string line;
		uint skip { head };
		
		Position position;
		Positions positions;
		while(std::getline(file, line)) {
			if (skip) {
				--skip;
				continue;
			}
			
			if (std::strcmp(filetype, "POSCAR") != 0) {
				throw std::runtime_error("Filetype not supported");
			}

			line = line.substr(2, line.length()); // strip leading whitespace
			std::string pos_x = line.substr(0, line.find("  "));
			std::string pos_y = line.substr(pos_x.length()+2, line.find("  "));
			std::string pos_z = line.substr(pos_x.length()+pos_y.length()+4, line.find(" "));
			
			positions.emplace_back(std::stod(pos_x), std::stod(pos_y), std::stod(pos_z));
		}
		
		file.close();
		return positions;
	}

	inline AtomPositions sortPositions(std::vector<Position> positions, const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
		AtomPositions atom_positions(N_Sr, N_Ti, N_O);
		std::ranges::copy(positions | std::views::take(N_Sr), std::back_inserter(atom_positions.SrPositions));
		std::ranges::copy(positions | std::views::drop(N_Sr) | std::views::take(N_Ti), std::back_inserter(atom_positions.TiPositions));
		std::ranges::copy(positions | std::views::drop(N_Sr+N_Ti) | std::views::take(N_O), std::back_inserter(atom_positions.OPositions));

		assert(atom_positions.SrPositions.size() == N_Sr);
		assert(atom_positions.TiPositions.size() == N_Ti);
		assert(atom_positions.OPositions.size() == N_O);

		return atom_positions;
	}
	
	inline std::vector<NearestNeighbors> getNearestNeighbors(const AtomPositions &atom_arr, const size_t n) {
		NearestNeighbors Sr_nearest_neighbors;
		NearestNeighbors Ti_nearest_neighbors;
		NearestNeighbors O_nearest_neighbors;
		
		std::ranges::for_each(atom_arr.SrPositions, [get_NN = helper::get_NN, &atom_arr, n, &Sr_nearest_neighbors](const Position &ref_atom) {
			helper::get_NN(atom_arr, n, Sr_nearest_neighbors, ref_atom);
		});

		std::ranges::for_each(atom_arr.TiPositions, [get_NN = helper::get_NN, &atom_arr, n, &Ti_nearest_neighbors](const Position &ref_atom) {
			get_NN(atom_arr, n, Ti_nearest_neighbors, ref_atom);
		});

		std::ranges::for_each(atom_arr.OPositions, [get_NN = helper::get_NN, &atom_arr, n, &O_nearest_neighbors](const Position &ref_atom) {
			get_NN(atom_arr, n, O_nearest_neighbors, ref_atom);
		});
		
		return std::vector<NearestNeighbors>({ Sr_nearest_neighbors, Ti_nearest_neighbors, O_nearest_neighbors });
	}

	// TODO correct NN calculations with PBC
}
