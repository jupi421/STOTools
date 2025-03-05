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
		AtomPositions atom_positions {};
		std::ranges::copy(positions | std::views::take(N_Sr), atom_positions.SrPositions.begin());
		std::ranges::copy(positions | std::views::drop(N_Sr) | std::views::take(N_Ti), atom_positions.TiPositions.begin());
		std::ranges::copy(positions | std::views::drop(N_Sr+N_Ti) | std::views::take(N_O), atom_positions.OPositions.begin());

		assert(atom_positions.SrPositions.size() == N_Sr);
		assert(atom_positions.TiPositions.size() == N_Ti);
		assert(atom_positions.OPositions.size() == N_O);

		return atom_positions;
	}
	
	inline auto get_n_nearest = [](const Position &atom1, const Positions &atom_arr, const size_t n) {
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

		Positions n_min_instances { n };
		for(size_t i=0; i<n; i++) {
			n_min_instances.emplace_back(atom_arr_ids.at(diff.at(i)));
		}

		return n_min_instances;
	};

	inline std::vector<AtomPositions> getNearestNeighbors(const AtomPositions &atom_arr, const size_t n) {
		AtomPositions Sr_nearest_neighbors;
		AtomPositions Ti_nearest_neighbors;
		AtomPositions O_nearest_neighbors;

		std::ranges::for_each(atom_arr.SrPositions, [&atom_arr, n](const Position &atom1) {
			Positions Sr_NN_Sr { get_n_nearest(atom1, atom_arr.SrPositions, n) };
			Positions Sr_NN_Ti { get_n_nearest(atom1, atom_arr.TiPositions, n) };
			Positions Sr_NN_O { get_n_nearest(atom1, atom_arr.OPositions, n) };

			Sr_nearest_neighbors.SrPositions.push_back(Sr_NN_Sr);
			Sr_nearest_neighbors.TiPositions.push_back(Ti_NN_Sr);
			Sr_nearest_neighbors.OPositions.push_back(O_NN_Sr);
		});

		std::ranges::for_each(atom_arr.TiPositions, [&atom_arr, n](const Position &atom1) {
			auto Ti_NN_Sr { get_n_nearest(atom1, atom_arr.SrPositions, n) };
			auto Ti_NN_Ti { get_n_nearest(atom1, atom_arr.TiPositions, n) };
			auto Ti_NN_O { get_n_nearest(atom1, atom_arr.OPositions, n) };
		});

		std::ranges::for_each(atom_arr.OPositions, [&atom_arr, n](const Position &atom1) {
			auto O_NN_Sr { get_n_nearest(atom1, atom_arr.SrPositions, n) };
			auto O_NN_Ti { get_n_nearest(atom1, atom_arr.TiPositions, n) };
			auto O_NN_O { get_n_nearest(atom1, atom_arr.OPositions, n) };
		});
	}
}
