#pragma once

#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <ranges>
#include <utility>
#include <optional>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

// TODO use omp to parallelise on cpu or cuda
// write different behavior for other filetypes (CONTCAR, XDATCAR, xyz, ...), calculate atom numbers??? 
namespace PolCalc {

using Position = Eigen::Vector3d;
using Vector = Eigen::Vector3d;
using Positions = std::vector<Position>;
using Vectors = std::vector<Vector>;
using NNIdSqDist = std::vector<std::pair<size_t, double>>; //all single species NN ids for a specific reference atom with squared distance


struct AtomPositions {

	Positions m_sr_positions {};
	Positions m_ti_positions {};
	Positions m_o_positions {};

	AtomPositions() {};
	AtomPositions(const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
		m_sr_positions.reserve(N_Sr);
		m_ti_positions.reserve(N_Ti);
		m_o_positions.reserve(N_O);
	}
};

struct NearestNeighborsByType {

	std::vector<NNIdSqDist> m_sr_nn_ids;
	std::vector<NNIdSqDist> m_ti_nn_ids;
	std::vector<NNIdSqDist> m_o_nn_ids;
};

std::vector<NNIdSqDist> getNearestNeighbors(const Positions &atoms, 
											const size_t n, 
											const bool is_same_atom_type, 
											const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt);


namespace helper {

static inline Position getCOM(const Positions &atom);

struct TetragonalUC {
	// tetragonal unit cell with COM at origin
	Positions m_atoms { };

	TetragonalUC() {
		m_atoms.reserve(8);

		constexpr double a { 3.905 };
		constexpr double c { 1.12 * a };

		Eigen::Vector3d ex = Eigen::Vector3d(1, 0, 0);
		Eigen::Vector3d ey = Eigen::Vector3d(0, 1, 0);
		Eigen::Vector3d ez = Eigen::Vector3d(0, 0, 1);

		m_atoms.emplace_back(-0.5*a*ex + 0.5*a*ey - 0.5*c*ez );
		m_atoms.emplace_back(0.5*a*ex + 0.5*a*ey - 0.5*c*ez );
		m_atoms.emplace_back(0.5*a*ex + 0.5*a*ey + 0.5*c*ez );
		m_atoms.emplace_back(-0.5*a*ex + 0.5*a*ey + 0.5*c*ez );
		m_atoms.emplace_back(-0.5*a*ex - 0.5*a*ey - 0.5*c*ez );
		m_atoms.emplace_back(0.5*a*ex - 0.5*a*ey - 0.5*c*ez );
		m_atoms.emplace_back(0.5*a*ex - 0.5*a*ey + 0.5*c*ez );
		m_atoms.emplace_back(-0.5*a*ex - 0.5*a*ey + 0.5*c*ez );
	}
};

struct LocalUC {
	Positions m_atoms { };
	Position m_COM { };
	double m_approx_tilt { };

	LocalUC(const Positions &atoms) {
		auto getLocalRefAxis = [&](const Positions &atoms) {
			// TODO overload function for NN for type Position
			std::vector<NNIdSqDist> nearest_neighbors { getNearestNeighbors(atoms, 1, true) };
			size_t nearest_neighbor_id { nearest_neighbors.at(0).at(0).first }; // get nearest neighbor for atom at atoms[0] 
			
			Vector local_axis { atoms.at(nearest_neighbor_id) - atoms.at(0) };
			if (local_axis[0] < 0) {
				local_axis *= -1; // make sure to take always the vector that points to the right to account for the correct rotaiton of the tetragon
			}

	 		local_axis[1] = 0; // ignore y shift

			return local_axis;
		};

		auto getSortedAngle = [this](const Positions &atoms, const Position &nearest_neighbor) { 

			auto getAngle = [this](const Position &r1, const Vector &x_axis) {
				double alpha_point { atan2(r1[0], r1[2]) };
				double alpha_ref_axis { atan2(x_axis[0], x_axis[2]) };

				double alpha { alpha_point - alpha_ref_axis };

				if (alpha < 0) {
					alpha += 2*M_PI;
				}

				m_approx_tilt = alpha_ref_axis;

				return alpha;
			};

			Positions atoms_zeroed { atoms };
			for (Position &atom : atoms_zeroed) {
				atom[1] = 0;
			}

			std::array<double, 4> angles;
			for (const size_t i : std::ranges::views::iota(4)) {
				angles.at(i) = getAngle(atoms_zeroed.at(i), nearest_neighbor);
			}

			std::array<size_t, 4> angle_ids;
			std::iota(angle_ids.begin(), angle_ids.end(), 0);

			std::ranges::sort(angle_ids, [&angles](const size_t id1, const size_t id2) {
				return angles.at(id1) < angles.at(id2);
			});

			return angles;
		};

		m_atoms.reserve(8);
		m_COM = helper::getCOM(atoms);

		Positions atoms_upper, atoms_lower;
		atoms_upper.reserve(4);
		atoms_lower.reserve(4);

		std::ranges::copy(atoms | std::ranges::views::filter([this](const Position &atom) { return atom[2] > this->m_COM[2]; }), std::back_inserter(atoms_upper));
		std::ranges::copy(atoms | std::ranges::views::filter([this](const Position &atom) { return atom[2] < this->m_COM[2]; }), std::back_inserter(atoms_lower));

		
		Vector local_x_axis { getLocalRefAxis(atoms_upper) }; // shouldn't really matter if the local x axis is taken from the upper half or lower half of the tetragon
		//Position nearest_neighbor_lower { getLocalAxis(atoms_lower) };
		
		std::array<double, 4> angles_upper { getSortedAngle(atoms_upper, local_x_axis) };
		std::array<double, 4> angles_lower { getSortedAngle(atoms_lower, local_x_axis) };

		m_atoms.push_back(atoms_upper.at(2));
		m_atoms.push_back(atoms_upper.at(3));
		m_atoms.push_back(atoms_upper.at(0));
		m_atoms.push_back(atoms_upper.at(1));
		m_atoms.push_back(atoms_lower.at(2));
		m_atoms.push_back(atoms_lower.at(3));
		m_atoms.push_back(atoms_lower.at(0));
		m_atoms.push_back(atoms_lower.at(1));
	}

};

inline Position getCOM(const Positions &atoms) {
	Vector COM = Eigen::Vector3d(0, 0, 0);

	for (const auto &atom : atoms) {
		COM += atom;
	}

	return COM/atoms.size();
}

inline Eigen::Vector3d convertCoordinates(Position &vector, const Eigen::Matrix3d &cell_matrix) {
	return cell_matrix*vector;
}

inline double getDistance(const Position &atom_1,
								 const Position &atom2,
								 const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt,
								 const bool take_root = false) {

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

	if (take_root) {
		return nearest_image.norm(); 
	}

	return nearest_image.squaredNorm();
}

inline NNIdSqDist findNNearest(const Position &reference_atom,
									  const Positions &atom_arr,
									  const size_t n,
									  const size_t reference_atom_id,
									  const bool is_same_atom_type,
									  const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt,
									  const bool sort = false) {

	std::vector<std::pair<size_t, double>> nearest_neighbors;
	nearest_neighbors.reserve(atom_arr.size());

	size_t idx { };
	for (const Position &other_atom : atom_arr) {
		nearest_neighbors.emplace_back(idx++, getDistance(reference_atom, other_atom, cell_matrix));
		//nearest_neighbors.emplace_back(std::make_pair(idx++, getDistance(reference_atom, other_atom, cell_matrix)));
	}

	if (is_same_atom_type) {
		nearest_neighbors.erase(nearest_neighbors.begin() + reference_atom_id);
	}

	std::ranges::nth_element(nearest_neighbors, nearest_neighbors.begin() + n, [](const auto &pair1, const auto &pair2){
		return pair1.second < pair2.second;
	});

	if (sort) {
		std::ranges::sort(nearest_neighbors.begin(), nearest_neighbors.begin() + n, [](const auto &p1, const auto &p2){
			return p1.second < p2.second;
		});
	}

	nearest_neighbors.resize(n);
	return nearest_neighbors;
}

// keep for testing purposes
inline void getNNAll(const AtomPositions &atom_arr,
							const size_t n,
							NearestNeighborsByType &NN_arr,
							const Position &ref_atom,
							const size_t ref_atom_id,
							const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt,
							const bool sort = true) {

	NNIdSqDist ref_atom_Sr_ids { findNNearest(ref_atom, atom_arr.m_sr_positions, n, ref_atom_id, true, cell_matrix, sort) };
	NNIdSqDist ref_atom_Ti_ids { findNNearest(ref_atom, atom_arr.m_ti_positions, n, ref_atom_id, false, cell_matrix, sort) };
	NNIdSqDist ref_atom_O_ids { findNNearest(ref_atom, atom_arr.m_o_positions, n, ref_atom_id, false, cell_matrix, sort) };

	NN_arr.m_sr_nn_ids.push_back(ref_atom_Sr_ids);
	NN_arr.m_ti_nn_ids.push_back(ref_atom_Ti_ids);
	NN_arr.m_o_nn_ids.push_back(ref_atom_O_ids);
}

inline void getNN(const Positions &atom_arr, 
				  const size_t n,  
				  std::vector<NNIdSqDist> &NN_arr, 
				  const Position &ref_atom, 
				  const size_t ref_atom_id, 
				  const bool same_atom_type, 
				  const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt, 
				  const bool sort = false) {

	NNIdSqDist ref_atom_NN { findNNearest(ref_atom, atom_arr, n, ref_atom_id, same_atom_type, cell_matrix, sort) };

	NN_arr.push_back(ref_atom_NN);
}

inline Vector getTranslationVec(const Position &pos1, const Position &pos2 = { 0, 0, 0 }) {
	return (pos1 - pos2);
}

static inline Position rotatePoint(const double angle, const Position &point) {
	Vector rot_axis { Eigen::Vector3d(0, 1, 0) };
	Eigen::AngleAxisd angle_axis { Eigen::AngleAxisd(angle, rot_axis) };
	Eigen::Matrix3d R { angle_axis.toRotationMatrix() };

	return R*point;
	
}

inline double getGradMSD(const double alpha, const Positions &pristine_UC, const Positions &local_UC) {
	double sin { std::sin(alpha) };
	double cos { std::cos(alpha) };
	Eigen::Matrix3d dR;

	dR << -sin, 0, cos,
			 0, 0, 0,
		  -cos, 0, -sin;


	double grad { };

	//make sure to calculate distance between correct atoms
	for (size_t i : std::ranges::views::iota(pristine_UC.size())) {
		grad += (local_UC.at(i) - rotatePoint(alpha, pristine_UC.at(i))).transpose()*(dR*pristine_UC.at(i));
	}

	return 2*grad;
}

inline double gradientDescent(double step_size, const TetragonalUC &pristine_UC, const LocalUC &local_UC) {
	double alpha { local_UC.m_approx_tilt };
	constexpr double threshold { 1e-12 };
	constexpr uint max_iter { 1000 };

	for (size_t i : std::ranges::views::iota(max_iter)) {
		double prev_alpha { alpha };
		alpha -= step_size * getGradMSD(alpha, pristine_UC.m_atoms, local_UC.m_atoms);

		double diff_alpha { std::abs(prev_alpha - alpha) };
		if (diff_alpha <= threshold) { 
			return alpha;
		}
	}

	return alpha;
}
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
	std::ranges::copy(positions | std::views::take(N_Sr), std::back_inserter(atom_positions.m_sr_positions));
	std::ranges::copy(positions | std::views::drop(N_Sr) | std::views::take(N_Ti), std::back_inserter(atom_positions.m_ti_positions));
	std::ranges::copy(positions | std::views::drop(N_Sr+N_Ti) | std::views::take(N_O), std::back_inserter(atom_positions.m_o_positions));

	assert(atom_positions.m_sr_positions.size() == N_Sr);
	assert(atom_positions.m_ti_positions.size() == N_Ti);
	assert(atom_positions.m_o_positions.size() == N_O);

	return atom_positions;
}

inline std::vector<NearestNeighborsByType> getNearestNeighborsAll(const AtomPositions &atom_arr, 
																  const size_t n, 
																  const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {
	NearestNeighborsByType Sr_nearest_neighbors;
	NearestNeighborsByType Ti_nearest_neighbors;
	NearestNeighborsByType O_nearest_neighbors;

	auto getNNPerType = [&](const Positions &atom_type_positions, NearestNeighborsByType &atom_type_nn_container) {
		size_t atom_id { 0 };
		for (const Position &ref_atom : atom_type_positions) {
			helper::getNNAll(atom_arr, n, atom_type_nn_container, ref_atom, atom_id++, cell_matrix);
		}
		atom_id = 0;
	};

	getNNPerType(atom_arr.m_sr_positions, Sr_nearest_neighbors);
	getNNPerType(atom_arr.m_ti_positions, Ti_nearest_neighbors);
	getNNPerType(atom_arr.m_o_positions, O_nearest_neighbors);

	return std::vector<NearestNeighborsByType>({ Sr_nearest_neighbors, Ti_nearest_neighbors, O_nearest_neighbors });
}

inline std::vector<NNIdSqDist> getNearestNeighbors(const Positions &atoms, 
												   const size_t n, 
												   const bool is_same_atom_type, 
												   const std::optional<Eigen::Matrix3d> &cell_matrix) {
	
	std::vector<NNIdSqDist> nearest_neighbors;

	size_t atom_id { 0 };
	for (const Position &ref_atom : atoms) {
		helper::getNN(atoms, n, nearest_neighbors, ref_atom, atom_id++, is_same_atom_type, cell_matrix);
	}

	return nearest_neighbors;
}

inline void getPolarization(const double step_size, const Positions &atoms, const std::vector<NNIdSqDist> &nearest_neighbors) {
	// atoms should be same elemenT as nearest_neighbors
	const helper::TetragonalUC tetragonal_UC { };

	Positions local_UC_atoms;
	local_UC_atoms.reserve(8);

	if (atoms.size() != nearest_neighbors.size()) {
		throw std::runtime_error("Mismatch in size of atoms and nearest_neighbors!");
	}

	for (const NNIdSqDist &pair_ref_atom_id_nn : nearest_neighbors) { // nearest neighbors for each center atom -> UC

		for (const auto &[nn_id, _]: pair_ref_atom_id_nn) {
			local_UC_atoms.emplace_back(atoms.at(nn_id));
		} 

		helper::LocalUC local_UC { helper::LocalUC(local_UC_atoms) };
		for (Position &atom : local_UC.m_atoms) {
			atom -= local_UC.m_COM;
		}

		double alpha { helper::gradientDescent(step_size, tetragonal_UC, local_UC) };

		Vectors displacements;
		displacements.reserve(8);

		Position rotatedPoint;
		for (size_t i : std::ranges::views::iota(8)) {
			rotatedPoint = helper::rotatePoint(alpha, tetragonal_UC.m_atoms.at(i));
			displacements.push_back(local_UC.m_atoms.at(i) - rotatedPoint);
		}

		// calculate BEC
	}
}
}
