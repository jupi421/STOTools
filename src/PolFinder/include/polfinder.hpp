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

namespace PolFinder {

	using Positions = std::vector<Eigen::Vector3d>;
	using Position = Eigen::Vector3d;
	using NNIDs = std::vector<size_t>;

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
		inline auto find_n_nearest = [](const Position &atom1, const Positions &atom_arr, const Eigen::Matrix3d &cell_matrix, const size_t n) {
			std::vector<double> dist { };
			dist.reserve(atom_arr.size());
			
			Eigen::Vector3d a0 { cell_matrix.row(0).transpose() };
			Eigen::Vector3d a1 { cell_matrix.row(1).transpose() };
			Eigen::Vector3d a2 { cell_matrix.row(2).transpose() };

			double Lx { a0.norm() };
			double Ly { a1.norm() };
			double Lz { a2.norm() };

			double Lx_relative { 1/Lx };
			double Ly_relative { 1/Ly };
			double Lz_relative { 1/Lz };
			
			std::ranges::for_each(atom_arr.begin(), atom_arr.end(), [&atom1, &dist, Lx, Ly, Lz, Lx_relative, Ly_relative, Lz_relative](const Position &atom2) {
				Eigen::Vector3d dr { atom2 - atom1 };
				double dx {	dr[0] - Lx*static_cast<int>(dr[0]*Lx_relative + 0.5) };
				double dy {	dr[1] - Ly*static_cast<int>(dr[1]*Ly_relative + 0.5) };
				double dz {	dr[2] - Lz*static_cast<int>(dr[2]*Lz_relative + 0.5) };

				dist.push_back(Eigen::Vector3d(dx, dy, dz).norm());
			});
			
			std::vector<size_t> atom_arr_ids(atom_arr.size());
			std::iota(atom_arr_ids.begin(), atom_arr_ids.end(), 0);
			
			std::ranges::sort(atom_arr_ids.begin(), atom_arr_ids.end(), [&dist](const size_t idx1, const size_t idx2) {
				return dist.at(idx1) < dist.at(idx2);
			});

			NNIDs n_min_instances;
			n_min_instances.reserve(n);
			std::ranges::copy_n(atom_arr_ids.begin(), n, std::back_inserter(n_min_instances));

			return n_min_instances;
		};

		inline auto get_NN = [](const AtomPositions &atom_arr, const size_t n, NearestNeighbors &NN_arr, const Position &ref_atom, const Eigen::Matrix3d &cell_matrix) {
			NNIDs ref_atom_Sr_ids { find_n_nearest(ref_atom, atom_arr.SrPositions, cell_matrix, n) };
			NNIDs ref_atom_Ti_ids { find_n_nearest(ref_atom, atom_arr.TiPositions, cell_matrix, n) };
			NNIDs ref_atom_O_ids { find_n_nearest(ref_atom, atom_arr.OPositions, cell_matrix, n) };

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
			line = line.substr(2, line.length()); // strip leading whitespace
			std::string pos_x = line.substr(0, line.find("  "));
			std::string pos_y = line.substr(pos_x.length()+2, line.find("  "));
			std::string pos_z = line.substr(pos_x.length()+pos_y.length()+4, line.find(" "));
			
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
		
		std::ranges::for_each(atom_arr.SrPositions, [get_NN = helper::get_NN, &atom_arr, n, &Sr_nearest_neighbors, &cell_matrix](const Position &ref_atom) {
			helper::get_NN(atom_arr, n, Sr_nearest_neighbors, ref_atom, cell_matrix);
		});

		std::ranges::for_each(atom_arr.TiPositions, [get_NN = helper::get_NN, &atom_arr, n, &Ti_nearest_neighbors, &cell_matrix](const Position &ref_atom) {
			get_NN(atom_arr, n, Ti_nearest_neighbors, ref_atom, cell_matrix);
		});

		std::ranges::for_each(atom_arr.OPositions, [get_NN = helper::get_NN, &atom_arr, n, &O_nearest_neighbors, &cell_matrix](const Position &ref_atom) {
			get_NN(atom_arr, n, O_nearest_neighbors, ref_atom, cell_matrix);
		});
		
		return std::vector<NearestNeighbors>({ Sr_nearest_neighbors, Ti_nearest_neighbors, O_nearest_neighbors });
	}

	// TODO correct NN calculations with PBC
}
