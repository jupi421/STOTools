#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <ranges>
#include <utility>
#include <optional>
#include <Eigen/Core>


#include <iostream>
#define print(x) std::cout << x << std::endl

// TODO use omp to parallelise on cpu or cuda
// write different behavior for other filetypes (CONTCAR, XDATCAR, xyz, ...), calculate atom numbers??? 
namespace PolCalc {

	using Position = Eigen::Vector3d;
	using Vector = Eigen::Vector3d;
	using Positions = std::vector<Position>;
	using NNIdSqDist = std::vector<std::pair<size_t, double>>; //all single species NN ids for a specific reference atom with squared distance

	struct TetragonalUC {
		// tetragonal unit cell with COM at origin
		Positions atoms {};

		TetragonalUC() {
			atoms.reserve(8);

			static constexpr double a { 3.905 };
			static constexpr double c { 1.12 * a };
		
			Eigen::Vector3d ex = Eigen::Vector3d(1, 0, 0);
			Eigen::Vector3d ey = Eigen::Vector3d(0, 1, 0);
			Eigen::Vector3d ez = Eigen::Vector3d(0, 0, 1);

			atoms.emplace_back(-0.5*a*ex + 0.5*a*ey - 0.5*c*ez );
			atoms.emplace_back(0.5*a*ex + 0.5*a*ey - 0.5*c*ez );
			atoms.emplace_back(0.5*a*ex + 0.5*a*ey + 0.5*c*ez );
			atoms.emplace_back(-0.5*a*ex + 0.5*a*ey + 0.5*c*ez );
			atoms.emplace_back(-0.5*a*ex - 0.5*a*ey - 0.5*c*ez );
			atoms.emplace_back(0.5*a*ex - 0.5*a*ey - 0.5*c*ez );
			atoms.emplace_back(0.5*a*ex - 0.5*a*ey + 0.5*c*ez );
			atoms.emplace_back(-0.5*a*ex - 0.5*a*ey + 0.5*c*ez );
		}
	};

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
		std::vector<NNIdSqDist> Sr_NN_ids;
		std::vector<NNIdSqDist> Ti_NN_ids;
		std::vector<NNIdSqDist> O_NN_ids;
	};

	namespace helper {
		static inline Eigen::Vector3d convertCoordinates(Position &vector, const Eigen::Matrix3d &cell_matrix) {
			return cell_matrix*vector;
		}

		static inline double getSquaredDistance(const Position &atom_1, const Position &atom2, const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt){
			Eigen::Vector3d dr { atom2 - atom_1 };
			

			double dx { std::abs(dr[0]) };
			double dy { std::abs(dr[1]) };
			double dz { std::abs(dr[2]) };

			dx -= static_cast<int>(dx + 0.5);
			dy -= static_cast<int>(dy + 0.5);
			dz -= static_cast<int>(dz + 0.5);
			
			Position nearest_image { Eigen::Vector3d(dx, dy, dz) };

			if (cell_matrix.has_value()) {
				nearest_image = helper::convertCoordinates(nearest_image, cell_matrix.value());
			}

			return nearest_image.norm();//squaredNorm();
		};

		static inline NNIdSqDist findNNearest(const Position &reference_atom, const Positions &atom_arr, const size_t n, const size_t reference_atom_id, const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {
			std::vector<std::pair<size_t, double>> nearest_neighbors;
			nearest_neighbors.reserve(atom_arr.size());
		
			size_t idx { };
			for (const Position &other_atom : atom_arr) {
				nearest_neighbors.emplace_back(std::make_pair(idx++, getSquaredDistance(reference_atom, other_atom, cell_matrix)));
			}

			nearest_neighbors.erase(nearest_neighbors.begin() + reference_atom_id);

			std::ranges::sort(nearest_neighbors, [](const auto &p1, const auto &p2){
				return p1.second < p2.second;
			});

			NNIdSqDist n_min_neighbors;
			n_min_neighbors.reserve(n);
			std::ranges::copy_n(nearest_neighbors.begin(), n, std::back_inserter(n_min_neighbors));

			return n_min_neighbors;
	
		};

		static inline void getNNAll(const AtomPositions &atom_arr, const size_t n, NearestNeighbors &NN_arr, const Position &ref_atom, const size_t ref_atom_id, const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {
			NNIdSqDist ref_atom_Sr_ids { findNNearest(ref_atom, atom_arr.SrPositions, n, ref_atom_id, cell_matrix) };
			NNIdSqDist ref_atom_Ti_ids { findNNearest(ref_atom, atom_arr.TiPositions, n, ref_atom_id, cell_matrix) };
			NNIdSqDist ref_atom_O_ids { findNNearest(ref_atom, atom_arr.OPositions, n, ref_atom_id, cell_matrix) };

			NN_arr.Sr_NN_ids.push_back(ref_atom_Sr_ids);
			NN_arr.Ti_NN_ids.push_back(ref_atom_Ti_ids);
			NN_arr.O_NN_ids.push_back(ref_atom_O_ids);
		};

		static inline void getNN(const Positions &atom_arr, const size_t n,  std::vector<NNIdSqDist> &NN_arr, const Position &ref_atom, const size_t ref_atom_id, const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {
			NNIdSqDist ref_atom_NN { findNNearest(ref_atom, atom_arr, n, ref_atom_id, cell_matrix) };

			NN_arr.push_back(ref_atom_NN);
		};

		static inline Vector getCOM(const Positions &atoms) {
			Vector COM = Eigen::Vector3d(0, 0, 0);

			for (const auto &atom : atoms) {
				COM += atom;
			}
			
			return COM;
		};

		static inline Vector getTranslationVec(const Position &pos1, const Position &pos2 = { 0, 0, 0 }) {
			return (pos1 - pos2);
		};
	}
	

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
	
	inline std::vector<NearestNeighbors> getNearestNeighborsAll(const AtomPositions &atom_arr, const size_t n, const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {
		NearestNeighbors Sr_nearest_neighbors;
		NearestNeighbors Ti_nearest_neighbors;
		NearestNeighbors O_nearest_neighbors;
		
		auto getNNPerType = [&](const Positions &atom_type_positions, NearestNeighbors &atom_type_nn_container) {
			size_t atom_id { 0 };
			for (const Position &ref_atom : atom_type_positions) {
				helper::getNNAll(atom_arr, n, atom_type_nn_container, ref_atom, atom_id++, cell_matrix);
			}
			atom_id = 0;
		};

		getNNPerType(atom_arr.SrPositions, Sr_nearest_neighbors);
		getNNPerType(atom_arr.TiPositions, Ti_nearest_neighbors);
		getNNPerType(atom_arr.OPositions, O_nearest_neighbors);
	 
		return std::vector<NearestNeighbors>({ Sr_nearest_neighbors, Ti_nearest_neighbors, O_nearest_neighbors });
	}

	inline std::vector<NNIdSqDist> getNearestNeighbors(const Positions &atoms, const size_t n, const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {
		std::vector<NNIdSqDist> nearest_neighbors;
		
		size_t atom_id { 0 };
		for (const Position &ref_atom : atoms) {
			helper::getNN(atoms, n, nearest_neighbors, ref_atom, atom_id++, cell_matrix);
		}
		
		return nearest_neighbors;
	}
		
	inline void getPolarization(const Positions &atoms, const std::vector<NNIdSqDist> &nearest_neighbors) {
		Positions local_UC;
		local_UC.reserve(8);

		for (const auto &pair_ref_atom_id_nn : nearest_neighbors) {

			for (const auto &[idx, dist] : pair_ref_atom_id_nn) {
				local_UC.push_back(atoms.at(idx));
			}

			TetragonalUC tetragonal_UC { };
			Vector COM { helper::getCOM(local_UC) };
			
			for (Position &atom : tetragonal_UC.atoms) {
				atom += COM;
			}
		}
	}
}
