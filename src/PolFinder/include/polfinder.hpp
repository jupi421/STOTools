#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <ranges>
#include <utility>
#include <numeric>
#include <Eigen/Core>

#include <iostream>
#define print(x) std::cout << x << std::endl

namespace PolFinder {

	using Positions = std::vector<Eigen::Vector3d>;
	using Position = Eigen::Vector3d;
	using NNIDs = std::vector<std::pair<size_t, double>>;

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
		std::vector<NNIDs> Sr_NN_ids;
		std::vector<NNIDs> Ti_NN_ids;
		std::vector<NNIDs> O_NN_ids;
	};

	namespace helper {
		inline auto get_direct_distance = [](const Position &atom_1, const Position &atom2, const size_t idx = 0){

			Eigen::Vector3d dr { atom2 - atom_1 };

			double dx { std::abs(dr[0]) };
			double dy { std::abs(dr[1]) };
			double dz { std::abs(dr[2]) };

			dx -= static_cast<int>(dx + 0.5);
			dy -= static_cast<int>(dy + 0.5);
			dz -= static_cast<int>(dz + 0.5);
			
			return Eigen::Vector3d(dx, dy, dz).norm();
		};

		inline auto find_n_nearest = [](const Position &reference_atom, const Positions &atom_arr, const size_t n, const size_t reference_atom_id) {
			std::vector<std::pair<size_t, double>> dist;
			dist.reserve(atom_arr.size());
		
			size_t idx { };
			std::ranges::for_each(atom_arr, [&dist, &reference_atom, &idx](const Position &other_atom){
				dist.emplace_back(std::make_pair(idx, get_direct_distance(reference_atom, other_atom)));
				idx++;
			});

			dist.erase(dist.begin() + reference_atom_id);

			std::ranges::sort(dist.begin(), dist.end(), [](const auto &p1, const auto &p2){
				return p1.second < p2.second;
			});

			NNIDs n_min_neighbors;
			n_min_neighbors.reserve(n);
			std::ranges::copy_n(dist.begin(), n, std::back_inserter(n_min_neighbors));

			return n_min_neighbors;
		};

		inline auto get_NN = [](const AtomPositions &atom_arr, const size_t n, NearestNeighbors &NN_arr, const Position &ref_atom, const size_t ref_atom_id) {
			NNIDs ref_atom_Sr_ids { find_n_nearest(ref_atom, atom_arr.SrPositions, n, ref_atom_id) };
			NNIDs ref_atom_Ti_ids { find_n_nearest(ref_atom, atom_arr.TiPositions, n, ref_atom_id) };
			NNIDs ref_atom_O_ids { find_n_nearest(ref_atom, atom_arr.OPositions, n, ref_atom_id) };

			NN_arr.Sr_NN_ids.push_back(ref_atom_Sr_ids);
			NN_arr.Ti_NN_ids.push_back(ref_atom_Ti_ids);
			NN_arr.O_NN_ids.push_back(ref_atom_O_ids);
		};
	}
	

	// write different behavior for other filetypes (CONTCAR, XDATCAR, xyz, ...), calculate atom numbers??? for end file and then parallelised for loop 
	inline Positions loadPosFromFile(std::string filename, uint head=0, long tail_start=-1, const char* filetype="POSCAR") {
		if (std::strcmp(filetype, "POSCAR") != 0) {
			throw std::runtime_error("Filetype not supported. (currently POSCAR only)");
		}

		std::ifstream file { filename };

		if (!file.is_open()) {
			throw std::runtime_error("Failed loading file!");
		}

		std::string line;
		uint skip { head };
		uint line_num { 0 };
		Position position;
		Positions positions;

		while(std::getline(file, line)) {

			++line_num;

			if (skip) {
				--skip;
				continue;
			}

			if (line_num == tail_start) {
				break;	
			}

			// TODO proper error handling if read something else than str
			// do read line with regex instead of hard coded whitespace length
			line = line.substr(2, line.length()); // strip leading whitespace
			std::string pos_x = line.substr(0, line.find("  "));
			std::string pos_y = line.substr(pos_x.length()+2, line.find("  "));
			std::string pos_z = line.substr(pos_x.length()+pos_y.length()+4, line.find(" "));
			
			//std::string pos_x = line.substr(0, line.find(" "));
			//std::string pos_y = line.substr(pos_x.length()+1, line.find(" "));
			//std::string pos_z = line.substr(pos_x.length()+pos_y.length()+2, line.length());
			positions.emplace_back(std::stod(pos_x), std::stod(pos_y), std::stod(pos_z));
		}
		
		file.close();
		return positions;
	}

	inline AtomPositions sortPositionsByType(std::vector<Position> positions, const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
		AtomPositions atom_positions(N_Sr, N_Ti, N_O);
		std::ranges::copy(positions | std::views::take(N_Sr), std::back_inserter(atom_positions.SrPositions));
		std::ranges::copy(positions | std::views::drop(N_Sr) | std::views::take(N_Ti), std::back_inserter(atom_positions.TiPositions));
		std::ranges::copy(positions | std::views::drop(N_Sr+N_Ti) | std::views::take(N_O), std::back_inserter(atom_positions.OPositions));

		assert(atom_positions.SrPositions.size() == N_Sr);
		assert(atom_positions.TiPositions.size() == N_Ti);
		assert(atom_positions.OPositions.size() == N_O);

		return atom_positions;
	}
	
	inline std::vector<NearestNeighbors> getNearestNeighbors(const AtomPositions &atom_arr, Eigen::Matrix3d &cell_matrix, const size_t n) {
		NearestNeighbors Sr_nearest_neighbors;
		NearestNeighbors Ti_nearest_neighbors;
		NearestNeighbors O_nearest_neighbors;
		
		size_t atom_id { 0 };
		std::ranges::for_each(atom_arr.SrPositions, [get_NN = helper::get_NN, &atom_arr, n, &Sr_nearest_neighbors, &atom_id](const Position &ref_atom) {
			helper::get_NN(atom_arr, n, Sr_nearest_neighbors, ref_atom, atom_id);
			atom_id++;
		});
		
		atom_id = 0;
		std::ranges::for_each(atom_arr.TiPositions, [get_NN = helper::get_NN, &atom_arr, n, &Ti_nearest_neighbors, &atom_id](const Position &ref_atom) {
			get_NN(atom_arr, n, Ti_nearest_neighbors, ref_atom, atom_id);
			atom_id++;
		});

		atom_id = 0;
		std::ranges::for_each(atom_arr.OPositions, [get_NN = helper::get_NN, &atom_arr, n, &O_nearest_neighbors, &atom_id](const Position &ref_atom) {
			get_NN(atom_arr, n, O_nearest_neighbors, ref_atom, atom_id);
			atom_id++;
		});
		
		return std::vector<NearestNeighbors>({ Sr_nearest_neighbors, Ti_nearest_neighbors, O_nearest_neighbors });
	}

	// TODO correct NN calculations with PBC
}
