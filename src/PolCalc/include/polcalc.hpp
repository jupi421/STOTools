#pragma once

#include <fstream>
#include <stdexcept>
#include <expected>
#include <print>
#include <string>
#include <vector>
#include <algorithm>
#include <ranges>
#include <utility>
#include <optional>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

// TODO use openACC to parallelise on cpu or cuda
// write different behavior for other filetypes (CONTCAR, XDATCAR, xyz, ...), calculate atom numbers??? 
namespace PolCalc {

using Position = Eigen::Vector3d;
using Positions = std::vector<Position>;
using Vector = Eigen::Vector3d;
using Vectors = std::vector<Vector>;
using NNIds = std::vector<std::pair<size_t, double>>; 

enum class DWType {
	HT, HH
};

enum class AtomType {
	Sr, Ti, O, Unknown
};

struct Atom {
	AtomType m_atom_type { AtomType::Unknown };
	Position m_position { Position::Zero() };

	Atom() {};
	Atom(AtomType atom_type, const Position& position) 
		: m_atom_type(atom_type), m_position(position)
	{}
};

using Atoms = std::vector<Atom>;

struct AtomPositions {

	Atoms m_Sr {};
	Atoms m_Ti {};
	Atoms m_O {};

	AtomPositions() {};
	AtomPositions(const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
		m_Sr.reserve(N_Sr);
		m_Ti.reserve(N_Ti);
		m_O.reserve(N_O);
	}
};

struct NearestNeighborsByType {

	std::vector<NNIds> m_sr_nn_ids;
	std::vector<NNIds> m_ti_nn_ids;
	std::vector<NNIds> m_o_nn_ids;
};

std::expected<std::vector<NNIds>, std::string> getNearestNeighbors(const Atoms& atoms, 
																   const Atoms& reference_atoms,
																   const size_t n, 
																   const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt);


namespace helper {

static inline Position getCOM(const Atoms &atom);

struct UnitCell {
	// tetragonal unit cell with COM at origin
	Atoms m_Sr_atoms { };
	Atoms m_Ti_atoms { };
	Atoms m_O_atoms { };
	Position m_COM { Position::Zero() };

	UnitCell() {};
	// apply cell matrix so that everything stays in direct coordinates
	UnitCell(Eigen::Matrix3d& cell_matrix) {
		m_Sr_atoms.reserve(8);

		constexpr double a { 3.905 };
		constexpr double c { 1.12 * a };

		Eigen::Vector3d ex = Eigen::Vector3d(1, 0, 0);
		Eigen::Vector3d ey = Eigen::Vector3d(0, 1, 0);
		Eigen::Vector3d ez = Eigen::Vector3d(0, 0, 1);

		m_Sr_atoms.emplace_back(AtomType::Sr, -0.5*a*ex + 0.5*a*ey - 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, 0.5*a*ex + 0.5*a*ey - 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, 0.5*a*ex + 0.5*a*ey + 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, -0.5*a*ex + 0.5*a*ey + 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, -0.5*a*ex - 0.5*a*ey - 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, 0.5*a*ex - 0.5*a*ey - 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, 0.5*a*ex - 0.5*a*ey + 0.5*c*ez );
		m_Sr_atoms.emplace_back(AtomType::Sr, -0.5*a*ex - 0.5*a*ey + 0.5*c*ez );
	}

	struct Displacements {
		Vectors m_Sr_displacements { };
		Vectors m_Ti_displacements { };
		Vectors m_O_displacements { };
	};

	Displacements operator-() const;
	void rotateUC();
};

struct LocalUC : UnitCell {

	enum class DWSide {
		left, right, Unknown
	};

	double m_tilt { };
	DWSide side { DWSide::Unknown };

	private:

	static Position minimumImage(const Position& pos) {
		return pos.array() - pos.array().round();
	}

	static Position directCoordinates(const Position& pos, const Position& ref) {
		Position temp { pos };
		temp += ref;
		for (size_t i : std::ranges::views::iota(3)) {
			temp[i] -= floor(temp[i]);
		}
		return temp;
	}

	static double getAngle(const Position& atom) {
		double angle { std::atan2(atom[2], atom[0]) };
		if (angle < 0) {
			angle += 2*M_PI;
		}
		return angle;
	}

	Vector getOrientation(DWType) const;

	public:

	LocalUC() {};
	explicit LocalUC(const Atoms& corners, const Atom& center, DWType DW_type, double DW_center_x = 0.5, double tolerance = 1e-3) { // explicit to prevent implicit type conversions
		
		if (corners.size() != 8) {
			throw std::runtime_error("LocalUC, Expected corners: 8, recieved: " + std::to_string(corners.size()));
		}

		m_Sr_atoms.reserve(8);
		
		Atoms corners_local { [&corners, &center](){
			Atoms temp { corners };
			for (Atom& atom : temp){
				atom.m_position = minimumImage(atom.m_position - center.m_position);
			}
			return temp;
		}()};

		Position COM_local = helper::getCOM(corners_local);
		m_COM = directCoordinates(COM_local, center.m_position);

		if (m_COM[0] < DW_center_x) {
			side = DWSide::left;
		}
		else if (m_COM[0] - DW_center_x < tolerance) { // consider thermal fluctuations
			side = DWSide::left;
		}
		else {
			side = DWSide::right;
		}

		std::vector<std::pair<Atom, double>> atoms_upper, atoms_lower;
		atoms_upper.reserve(4);
		atoms_lower.reserve(4);

		for (const Atom& corner : corners_local) {
			double angle { getAngle(corner.m_position - COM_local) };
			(corner.m_position[1] < COM_local[1] ? atoms_lower : atoms_upper).emplace_back(corner, angle);
		}

		if (atoms_upper.size() != 4) {
			throw std::runtime_error("LocalUC, atoms_upper requires 4 elements, got " + std::to_string(atoms_upper.size()));
		}
		else if (atoms_lower.size() != 4) {
			throw std::runtime_error("LocalUC, atoms_lower requires 4 elements, got " + std::to_string(atoms_upper.size()));
		}

		for (auto& pair : atoms_upper) {
			pair.first.m_position = directCoordinates(pair.first.m_position, center.m_position);
		}
		for (auto& pair : atoms_lower) {
			pair.first.m_position = directCoordinates(pair.first.m_position, center.m_position);
		}

		auto sort_angles = [](auto& arr) { 
			std::ranges::sort(arr, [](auto& pair1, auto& pair2) { 
				return pair1.second < pair2.second;
			});};

		sort_angles(atoms_upper);
		sort_angles(atoms_lower);

		if (side == DWSide::left) {
			m_Sr_atoms.push_back(atoms_upper.at(2).first);
			m_Sr_atoms.push_back(atoms_upper.at(3).first);
			m_Sr_atoms.push_back(atoms_upper.at(0).first);
			m_Sr_atoms.push_back(atoms_upper.at(1).first);
			m_Sr_atoms.push_back(atoms_lower.at(2).first);
			m_Sr_atoms.push_back(atoms_lower.at(3).first);
			m_Sr_atoms.push_back(atoms_lower.at(0).first);
			m_Sr_atoms.push_back(atoms_lower.at(1).first);
		}
		else if (side == DWSide::right) {
			m_Sr_atoms.push_back(atoms_upper.at(3).first);
			m_Sr_atoms.push_back(atoms_upper.at(0).first);
			m_Sr_atoms.push_back(atoms_upper.at(1).first);
			m_Sr_atoms.push_back(atoms_upper.at(2).first);
			m_Sr_atoms.push_back(atoms_lower.at(3).first);
			m_Sr_atoms.push_back(atoms_lower.at(0).first);
			m_Sr_atoms.push_back(atoms_lower.at(1).first);
			m_Sr_atoms.push_back(atoms_lower.at(2).first);
		}
	}

	void transformToMinimumImage();
	void transformToDirect();
	void rotateUC() = delete;

};

inline Position getCOM(const Atoms &atoms) {
	Vector COM = Vector::Zero();

	for (const auto &atom : atoms) {
		COM += atom.m_position;
	}

	return COM/atoms.size();
}

inline Eigen::Vector3d convertCoordinates(Position &vector, const Eigen::Matrix3d &cell_matrix) {
	return cell_matrix*vector;
}

inline double getDistance(const Atom& atom1, 
						  const Atom& atom2, 
						  const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt, 
						  const bool norm = false) {

	Vector dr { atom2.m_position - atom1.m_position };
	dr.array() -= dr.array().round();

	Position minimum_image_distance { dr };

	if (cell_matrix) {
		minimum_image_distance = helper::convertCoordinates(minimum_image_distance, cell_matrix.value());
	}

	if (norm) {
		return minimum_image_distance.norm(); 
	}

	return minimum_image_distance.squaredNorm();
}

inline std::expected<NNIds, std::string> findNearestN(const Atom& reference_atom, 
														   const Atoms& atom_arr, 
														   const size_t n, 
														   const std::optional<size_t> exclude_idx,
														   const std::optional<Eigen::Matrix3d>& cell_matrix = std::nullopt, 
														   const bool sort = false) {

	if (n > atom_arr.size()) {
		return std::unexpected("n cannot exceed size of atom_arr!");
	}
	else if (n == 0) {
		return std::unexpected("n cannot be 0!");
	}

	if (exclude_idx) {
		if (n > atom_arr.size()-1 ) {
			return std::unexpected("n cannot exceed size of atom_arr!");
		}
	}

	std::vector<std::pair<size_t, double>> nearest_neighbors;
	nearest_neighbors.reserve(atom_arr.size());
	size_t idx { };

	for (const Atom& other_atom : atom_arr) {
		if (exclude_idx && idx == exclude_idx.value()) {
			idx++;
			continue;
		}

		nearest_neighbors.emplace_back(idx++, getDistance(reference_atom, other_atom, cell_matrix));
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

inline Vector getTranslationVec(const Position &pos1, const Position &pos2 = { 0, 0, 0 }) {
	return (pos1 - pos2);
}

static inline Position rotatePoint(const double angle, const Position &point) {
	Vector rot_axis { Eigen::Vector3d(0, 1, 0) };
	Eigen::AngleAxisd angle_axis { Eigen::AngleAxisd(angle, rot_axis) };
	Eigen::Matrix3d R { angle_axis.toRotationMatrix() };

	return R*point;
	
}

inline double getGradMSD(const double alpha, const Atoms &pristine_UC, const Atoms &local_UC) {
	double sin { std::sin(alpha) };
	double cos { std::cos(alpha) };
	Eigen::Matrix3d dR;

	dR << -sin, 0, cos,
			 0, 0, 0,
		  -cos, 0, -sin;


	double grad { };

	//make sure to calculate distance between correct atoms
	for (size_t i : std::ranges::views::iota(pristine_UC.size())) {
		grad += (local_UC.at(i).m_position - rotatePoint(alpha, pristine_UC.at(i).m_position)).transpose()*(dR*pristine_UC.at(i).m_position);
	}

	return 2*grad;
}

inline double gradientDescent(double step_size, const UnitCell &pristine_UC, const LocalUC &local_UC) {
	double alpha { local_UC.m_tilt };
	constexpr double threshold { 1e-12 };
	constexpr uint max_iter { 1000 };

	for (size_t i : std::ranges::views::iota(max_iter)) {
		double prev_alpha { alpha };
		alpha -= step_size * getGradMSD(alpha, pristine_UC.m_Sr_atoms, local_UC.m_Sr_atoms);

		double diff_alpha { std::abs(prev_alpha - alpha) };
		if (diff_alpha <= threshold) { 
			return alpha;
		}
	}

	return alpha;
}
}


inline Positions loadPosFromFile(std::string filename,
                                 uint head = 0,
                                 long tail_start = -1,
                                 const char* filetype = "POSCAR")
{
    if (std::strcmp(filetype, "POSCAR") != 0) {
        throw std::runtime_error("Filetype not supported. (currently POSCAR only)");
    }

    std::ifstream file{filename};
    if (!file.is_open()) {
        throw std::runtime_error("Failed loading file: " + filename);
    }

    std::string line;
    // skip header lines
    for (uint i = 0; i < head && std::getline(file, line); ++i) { /* skip */ }

    Positions positions;
    long line_num = static_cast<long>(head);

    while (std::getline(file, line)) {
        ++line_num;
        if (tail_start > 0 && line_num >= tail_start) break;

        // Try to read 3 numbers from the line, ignoring whitespace
        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) {
            // Non-coordinate line (blank / comments) → skip quietly
            // If you prefer hard failure, replace with:
            // throw std::runtime_error("Parse error at line " + std::to_string(line_num) + ": '" + line + "'");
            continue;
        }
        positions.emplace_back(x, y, z);
    }

    return positions;
}

//inline Positions loadPosFromFile(std::string filename, uint head=0, long tail_start=-1, const char* filetype="POSCAR") {
//	if (std::strcmp(filetype, "POSCAR") != 0) {
//		throw std::runtime_error("Filetype not supported. (currently POSCAR only)");
//	}
//
//	std::ifstream file { filename };
//
//	if (!file.is_open()) {
//		throw std::runtime_error("Failed loading file!");
//	}
//
//	std::string line;
//	uint skip { head };
//	uint line_num { 0 };
//	Positions positions;
//
//	while(std::getline(file, line)) {
//
//		++line_num;
//
//		if (skip) {
//			--skip;
//			continue;
//		}
//
//		if (line_num == tail_start) {
//			break;	
//		}
//
//		// TODO proper error handling if read something else than str
//		// do read line with regex instead of hard coded whitespace length
//		line = line.substr(2, line.length()); // strip leading whitespace
//		std::string pos_x = line.substr(0, line.find("  "));
//		std::string pos_y = line.substr(pos_x.length()+2, line.find("  "));
//		std::string pos_z = line.substr(pos_x.length()+pos_y.length()+4, line.find(" "));
//
//		//std::string pos_x = line.substr(0, line.find(" "));
//		//std::string pos_y = line.substr(pos_x.length()+1, line.find(" "));
//		//std::string pos_z = line.substr(pos_x.length()+pos_y.length()+2, line.length());
//		positions.emplace_back(std::stod(pos_x), std::stod(pos_y), std::stod(pos_z));
//	}
//
//	file.close();
//	return positions;
//}

inline std::expected<AtomPositions, std::string> sortPositionsByType(const Positions& positions, const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
	AtomPositions atom_positions(N_Sr, N_Ti, N_O);
	std::ranges::copy(positions 
				   | std::views::transform([](const Position& positions){ return Atom(AtomType::Sr, positions); }) 
				   | std::views::take(N_Sr), std::back_inserter(atom_positions.m_Sr));
	std::ranges::copy(positions 
				   | std::views::transform([](const Position& positions){ return Atom(AtomType::Ti, positions); }) 
				   | std::views::drop(N_Sr) | std::views::take(N_Ti), std::back_inserter(atom_positions.m_Ti));
	std::ranges::copy(positions 
				   | std::views::transform([](const Position& positions){ return Atom(AtomType::O, positions); }) 
				   | std::views::drop(N_Sr+N_Ti) 
				   | std::views::take(N_O), std::back_inserter(atom_positions.m_O));

	if (atom_positions.m_Sr.size() != N_Sr) {
		return std::unexpected("m_Sr positions and N_Sr differ");
	};
	if (atom_positions.m_Ti.size() != N_Ti) {
		return std::unexpected("m_Ti positions and N_Ti differ");
	}
	if (atom_positions.m_O.size() != N_O) {
		return std::unexpected("m_O positions and N_O differ");
	}

	return atom_positions;
}

inline std::expected<std::vector<NNIds>, std::string> getNearestNeighbors(const Atoms& atoms, 
																		  const Atoms& ref_atoms,
																		  const size_t n, 
																		  const std::optional<Eigen::Matrix3d>& cell_matrix,
																		  bool sort = false) {
	
	std::vector<NNIds> nearest_neighbors;
	nearest_neighbors.reserve(ref_atoms.size());

	size_t atom_id { 0 };
	for (const Atom& ref_atom : ref_atoms) {

		std::optional<size_t> exclude_idx { std::nullopt };
		if (&atoms == &ref_atoms) {
			exclude_idx = atom_id;
		}

		//auto res = helper::getNN(atoms, n, nearest_neighbors, ref_atom, exclude_idx, cell_matrix);
		auto res = helper::findNearestN(ref_atom, atoms, n, exclude_idx, cell_matrix, sort) 
			.transform([&nearest_neighbors](auto&& res) { nearest_neighbors.push_back(std::move(res)); });

		if (!res) {
			return std::unexpected("On iteration " + std::to_string(atom_id) + ", in findNearestN: " + res.error());
		}

		atom_id++;
	}

	return nearest_neighbors;
}

inline void getPolarization(const double step_size, const Atoms &atoms, const std::vector<NNIds> &nearest_neighbors) {
	// atoms should be same element as nearest_neighbors
	const helper::UnitCell tetragonal_UC { };

	Atoms local_UC_atoms;
	local_UC_atoms.reserve(8);

	if (atoms.size() != nearest_neighbors.size()) {
		throw std::runtime_error("Mismatch in size of atoms and nearest_neighbors!");
	}

	for (const NNIds &pair_ref_atom_id_nn : nearest_neighbors) { // nearest neighbors for each center atom -> UC

		for (const auto &[nn_id, _]: pair_ref_atom_id_nn) {
			local_UC_atoms.emplace_back(atoms.at(nn_id));
		} 

		helper::LocalUC local_UC { helper::LocalUC(local_UC_atoms) };
		for (Atom &atom : local_UC.m_Sr_atoms) {
			atom.m_position -= local_UC.m_COM;
		}

		double alpha { helper::gradientDescent(step_size, tetragonal_UC, local_UC) };

		Vectors displacements;
		displacements.reserve(8);

		Position rotatedPoint;
		for (size_t i : std::ranges::views::iota(8)) {
			rotatedPoint = helper::rotatePoint(alpha, tetragonal_UC.m_Sr_atoms.at(i).m_position);
			displacements.push_back(local_UC.m_Sr_atoms.at(i).m_position - rotatedPoint);
		}

		// calculate BEC
	}
}
}
