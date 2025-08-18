#pragma once

#include <iostream>
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
	HT, HH, APB
};

enum class AtomType {
	Sr, Ti, O, Unknown
};

struct Atom {
	AtomType m_atom_type { AtomType::Unknown };
	Position m_position { Position::Zero() };

	Atom() = default;
	Atom(AtomType atom_type, const Position& position) 
		: m_atom_type(atom_type), m_position(position)
	{}
};

using Atoms = std::vector<Atom>;

struct AtomPositions {

	Atoms m_Sr {};
	Atoms m_Ti {};
	Atoms m_O {};

	AtomPositions() = default;
	AtomPositions(const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
		m_Sr.reserve(N_Sr);
		m_Ti.reserve(N_Ti);
		m_O.reserve(N_O);
	}
};


std::expected<std::vector<NNIds>, std::string> getNearestNeighbors(const Atoms& atoms, 
																   const Atoms& reference_atoms,
																   const size_t n, 
																   const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt);


namespace helper {

inline Eigen::Vector3d convertCoordinates(const Position &pos, const Eigen::Matrix3d &cell_matrix);

static inline Position getCOM(const Atoms &atom);

	class UnitCell {
	// tetragonal unit cell with COM at origin
public:
	// A and O contain pairs of opposite atoms (wrt y axis)
	std::vector<std::pair<Atom, Atom>> m_A_cart_nopbc { };
	Atom m_B_cart_nopbc { };
	std::vector<std::pair<Atom, Atom>> m_O_cart_nopbc { };
	Position m_COM_cart_nopbc { Position::Zero() };

	UnitCell() {};

	UnitCell(AtomType type_A, AtomType type_B, AtomType type_O, short O_rot_sign) {
		m_A_cart_nopbc.reserve(8);

		constexpr double a { 3.905 };
		constexpr double c { a }; 

		Eigen::Vector3d ex = Eigen::Vector3d(1, 0, 0);
		Eigen::Vector3d ey = Eigen::Vector3d(0, 1, 0);
		Eigen::Vector3d ez = Eigen::Vector3d(0, 0, 1);

		auto fill_pristine = [&](AtomType type, Position&& pos){
			Atom first { type, pos };
			pos[1] *= -1;
			Atom second { type, pos}; // mirroring first at xz plane
			m_A_cart_nopbc.emplace_back(std::move(first), std::move(second));
		};

		fill_pristine(type_A, - 0.5*a*ex + 0.5*a*ey - 0.5*c*ez);
		fill_pristine(type_A, 0.5*a*ex + 0.5*a*ey - 0.5*c*ez);
		fill_pristine(type_A, 0.5*a*ex + 0.5*a*ey + 0.5*c*ez);
		fill_pristine(type_A, -0.5*a*ex + 0.5*a*ey + 0.5*c*ez);
	}

	struct Displacements {
		Vectors m_A_displacements { };
		Vectors m_B_displacements { };
		Vectors m_O_displacements { };
	};

	Displacements operator-() const;
	void rotateUC();
};

class LocalUC : UnitCell {

public:
	enum class DWSide {
		left, right, center, Unknown
	};

	std::vector<std::pair<Atom, Atom>> m_A_direct_pbc { };
	Atom m_B_direct_pbc { };
	std::vector<std::pair<Atom, Atom>> m_O_direct_pbc { };
	std::optional<double> m_angle { };
	DWSide side { DWSide::Unknown };

private:
	static inline std::optional<double> m_right_init_angle;
	static inline std::optional<double> m_left_init_angle;
	Eigen::Matrix3d m_metric;

	struct OPairs {
		int m_first_id { };
		int m_second_id { };
		double m_cos { };
		
		OPairs(int first_id, int second_id, double sq_norm)
			: m_first_id(first_id), m_second_id(second_id), m_cos(sq_norm)
		{}
	};

	static Position minimumImage(const Position& pos) {
		return pos.array() - pos.array().round();
	}

	static double wrapDirectCoordinates(double x) {
		return x - floor(x);
	}

	static Position wrapDirectCoordinates(const Position& pos) {
		Position temp { pos };
		for (size_t i { 0 }; i < 3; i++) {
			temp[i] = wrapDirectCoordinates(temp[i]);
		}
		return temp;
	}
	
	static void wrapDirectCoordinates(std::vector<std::pair<Atom, double>>& atoms, Position&& ref) {
		for (auto& pair : atoms) {
			pair.first.m_position = wrapDirectCoordinates(pair.first.m_position + ref);
		}
	}

	static void wrapDirectCoordinates(std::vector<std::pair<Atom, Atom>>& atoms, const Position& ref) {
		for (auto& pair : atoms) {
			pair.first.m_position = wrapDirectCoordinates(pair.first.m_position + ref);
			pair.second.m_position = wrapDirectCoordinates(pair.second.m_position + ref);
		}
	}

	Position get_cart_pos_nowrap(const Atom& atom, const Atom& ref, const Eigen::Matrix3d& cell_matrix) {
		Position pos_unwrapped { ref.m_position + minimumImage( atom.m_position - ref.m_position ) };
		return convertCoordinates(pos_unwrapped, cell_matrix);
	};

	static double getAngle(const Position& atom) {
		double angle { std::atan2(atom[2], atom[0]) };
		if (angle < 0) {
			angle += 2*M_PI;
		}
		return angle;
	}

	void setDomain(const Position& ref, double DW_center_x, double tolerance) {
		double distance_from_DW = wrapDirectCoordinates(ref)[0] - DW_center_x;
		distance_from_DW -= round(distance_from_DW);

		if (ref.x() < DW_center_x+tolerance && ref.x() > DW_center_x-tolerance) {
			side = DWSide::center;
		}
		else if (ref.x() > DW_center_x) {
			side = DWSide::right;
		}
		else {
			side = DWSide::left;
		}
	}
	
	void rebalancePlanes(std::vector<std::pair<Atom, double>>& atoms_upper, std::vector<std::pair<Atom, double>>& atoms_lower, size_t recursion_depth=0) {
		if (recursion_depth > 8) {
			return;
		}
		else if (atoms_upper.size() == 4 && atoms_lower.size() == 4) {
			return;
		}

		if ((atoms_upper.size() + atoms_lower.size() != 8)) {
			throw std::runtime_error("LocalUC::LocalUC incorrect number of atoms");
		}
		else if (atoms_upper.size() > 4 && atoms_lower.size() < 4) {
			auto lowest_upper { std::ranges::min_element(atoms_upper, [](const auto& lhs, const auto& rhs){
				return lhs.first.m_position[1] < rhs.first.m_position[1];
			}) };
			atoms_lower.push_back(*lowest_upper);
			atoms_upper.erase(lowest_upper);
		}
		else if (atoms_upper.size() < 4 && atoms_lower.size() > 4) {
			auto highest_lower { std::ranges::max_element(atoms_lower, [](const auto& lhs, const auto& rhs){
				return lhs.first.m_position[1] < rhs.first.m_position[1];
			}) };
			atoms_upper.push_back(*highest_lower);
			atoms_lower.erase(highest_lower);
		}
		 
		return rebalancePlanes(atoms_upper, atoms_lower, ++recursion_depth);
	}

	std::vector<std::pair<Atom, Atom>> findOppositeO(const Atoms& atoms, double cos_tolerance = 0.1) {
		std::vector<std::pair<Atom, Atom>> matches;
		size_t matches_size { atoms.size() / 2 };
		matches.reserve(matches_size);
		for (size_t i { 0 }; i < atoms.size(); i++) {
			for (size_t j { 0 }; j < i; j++) {
				Position pos1 { atoms[i].m_position };
				Position pos2 { atoms[j].m_position };
				
				double cos { pos1.dot(m_metric*pos2) / sqrt(pos1.dot(m_metric*pos1) * pos2.dot(m_metric*pos2)) };
				if (1+cos < cos_tolerance) {
					matches.emplace_back(atoms[i], atoms[j]);
				}
			}
		}
		if (matches.size() != matches_size) {
			throw std::runtime_error("Matches are " + std::to_string(matches.size()) + " but should be " + std::to_string(matches_size));
		}
		return matches;
	}

public:
	Vector getOrientation(DWType) const;

	explicit LocalUC(const Atoms& A, const Atom& B, const Atoms& O, DWType DW_type, const Eigen::Matrix3d& cell_matrix, double DW_center_x = 0.5, double tolerance = 1e-3) 
		: m_B_direct_pbc(B), m_metric(cell_matrix.transpose() * cell_matrix)
	{
		if (A.size() != 8) {
			throw std::runtime_error("LocalUC, Expected corners: 8, recieved: " + std::to_string(A.size()));
		}
		else if (O.size() != 6) {
			throw std::runtime_error("LocalUC, Expected O: 6, recieved: " + std::to_string(O.size()));
		}
		
		Atoms A_rel_to_B { 
			[&A, &B](){
				Atoms temp { A };
				for (Atom& atom : temp){
					atom.m_position = minimumImage(atom.m_position - B.m_position);
				}
				return temp;
			}()
		};

		Position COM_rel_to_B = helper::getCOM(A_rel_to_B);

		setDomain(B.m_position + COM_rel_to_B, DW_center_x, tolerance);

		std::vector<std::pair<Atom, double>> atoms_upper, atoms_lower;
		atoms_upper.reserve(4);
		atoms_lower.reserve(4);

		// get angles and split corners into upper and lower
		for (const Atom& corner : A_rel_to_B) {
			Atom corner_rel_to_COM { corner.m_atom_type, minimumImage(corner.m_position - COM_rel_to_B) };
			double angle { getAngle(cell_matrix*corner_rel_to_COM.m_position) };
			(corner_rel_to_COM.m_position[1] < 0 ? atoms_lower : atoms_upper).emplace_back(corner_rel_to_COM, angle);
		}

		rebalancePlanes(atoms_upper, atoms_lower);

		// wrap from direct atoms COM centered back to direct coordinates with pbc
		wrapDirectCoordinates(atoms_upper, B.m_position + COM_rel_to_B);
		wrapDirectCoordinates(atoms_lower, B.m_position + COM_rel_to_B);

		auto sort_angles = [](auto& arr) { 
			std::ranges::sort(arr, [](auto& pair1, auto& pair2) { 
				return pair1.second < pair2.second;
			});};

		sort_angles(atoms_upper);
		sort_angles(atoms_lower);

		// fill A and O arrays
		auto fill_direct_A = [&](auto&& upper, auto&& lower){
			for (size_t i { 0 }; i < upper.size(); i++) {
				Atom first { std::move(upper[i].first)};
				Atom second { std::move(lower[i].first)};
				m_A_direct_pbc.emplace_back(std::move(first), std::move(second));
			}
		};

		m_A_direct_pbc.reserve(4);
		fill_direct_A(atoms_upper, atoms_lower);
		
		// sort O top, bottom, ....
		Atoms O_B_centered { 
			[&](){
				Atoms temp { O };
				for (Atom& atom : temp){
					atom.m_position = minimumImage(atom.m_position - B.m_position);
				}
				return temp; 
			}()
		};

		std::vector<std::pair<Atom, Atom>> O_B_centered_pairs { findOppositeO(O_B_centered) };
		// wrap O back to direct with pbc
		wrapDirectCoordinates(O_B_centered_pairs, B.m_position);
		m_O_direct_pbc = O_B_centered_pairs;
		
		// get cartesian coordinates without pbc
		m_A_cart_nopbc.reserve(4);
		m_O_cart_nopbc.reserve(3);

		auto fill_cart = [&](const auto& direct, auto& cart) {
			for (const auto& pair : direct) {
				Atom first { pair.first.m_atom_type, get_cart_pos_nowrap(pair.first, B, cell_matrix) };
				Atom second { pair.second.m_atom_type, get_cart_pos_nowrap(pair.second, B, cell_matrix) };
				cart.emplace_back(std::move(first), std::move(second));
			}; };

		// save in cartesian without pbc
		fill_cart(m_A_direct_pbc, m_A_cart_nopbc);
		fill_cart(m_O_direct_pbc, m_O_cart_nopbc);
		m_B_cart_nopbc = Atom(m_B_direct_pbc.m_atom_type, get_cart_pos_nowrap(B, B, cell_matrix));
		m_COM_cart_nopbc = convertCoordinates(B.m_position + COM_rel_to_B, cell_matrix);
	}

	std::expected<double, std::string> get_init_angle() {
		if (!m_left_init_angle || !m_right_init_angle) {
			return std::unexpected("Angle not initialized!");
		}
		else if (side == DWSide::right) {
			return m_right_init_angle.value();
		}
		else if (side == DWSide::left) {
			return m_left_init_angle.value();
		}
		else {
			return m_left_init_angle.value();
		}
	}

	//explicit LocalUC(const Atoms& A, const Atom& B, const Atoms& O, DWType DW_type, const Eigen::Matrix3d& cell_matrix, double DW_center_x = 0.5, double tolerance = 1e-3) {
	//	if (A.size() != 8) {
	//		throw std::runtime_error("LocalUC, Expected corners: 8, recieved: " + std::to_string(A.size()));
	//	}
	//	else if (O.size() != 6) {
	//		throw std::runtime_error("LocalUC, Expected O: 6, recieved: " + std::to_string(O.size()));
	//	}
	//	
	//	Atoms corners_rel_to_B { [&A, &B](){
	//		Atoms temp { A };
	//		for (Atom& atom : temp){
	//			atom.m_position = minimumImage(atom.m_position - B.m_position);
	//		}
	//		return temp;
	//	}()};

	//	Position COM_rel_to_B = helper::getCOM(corners_rel_to_B);

	//	setDomain(wrapDirectCoordinates(B.m_position + COM_rel_to_B), DW_center_x, tolerance);

	//	std::vector<std::pair<Atom, double>> atoms_upper, atoms_lower;
	//	atoms_upper.reserve(4);
	//	atoms_lower.reserve(4);

	//	for (const Atom& corner : corners_rel_to_B) {
	//		double angle { getAngle(minimumImage(corner.m_position - COM_rel_to_B)) };
	//		(corner.m_position[1] < COM_rel_to_B[1] ? atoms_lower : atoms_upper).emplace_back(corner, angle);
	//	}

	//	if (atoms_upper.size() != 4) {
	//		throw std::runtime_error("LocalUC, atoms_upper requires 4 elements, got " + std::to_string(atoms_upper.size()));
	//	}
	//	else if (atoms_lower.size() != 4) {
	//		throw std::runtime_error("LocalUC, atoms_lower requires 4 elements, got " + std::to_string(atoms_upper.size()));
	//	}

	//	for (auto& pair : atoms_upper) {
	//		pair.first.m_position = wrapDirectCoordinates(pair.first.m_position + B.m_position);
	//	}
	//	for (auto& pair : atoms_lower) {
	//		pair.first.m_position = wrapDirectCoordinates(pair.first.m_position + B.m_position);
	//	}

	//	auto sort_angles = [](auto& arr) { 
	//		std::ranges::sort(arr, [](auto& pair1, auto& pair2) { 
	//			return pair1.second < pair2.second;
	//		});};

	//	sort_angles(atoms_upper);
	//	sort_angles(atoms_lower);

	//	if (side == DWSide::left) {
	//		m_A_direct_pbc.push_back(atoms_upper.at(2).first);
	//		m_A_direct_pbc.push_back(atoms_upper.at(3).first);
	//		m_A_direct_pbc.push_back(atoms_upper.at(0).first);
	//		m_A_direct_pbc.push_back(atoms_upper.at(1).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(2).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(3).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(0).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(1).first);
	//	}
	//	else if (side == DWSide::right) {
	//		m_A_direct_pbc.push_back(atoms_upper.at(3).first);
	//		m_A_direct_pbc.push_back(atoms_upper.at(0).first);
	//		m_A_direct_pbc.push_back(atoms_upper.at(1).first);
	//		m_A_direct_pbc.push_back(atoms_upper.at(2).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(3).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(0).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(1).first);
	//		m_A_direct_pbc.push_back(atoms_lower.at(2).first);
	//	}
	//	
	//	auto get_cart_pos_nowrap = [&](const Atom& atom) {
	//		Position pos_unwrapped { B.m_position + minimumImage( atom.m_position - B.m_position ) };
	//		return convertCoordinates(pos_unwrapped, cell_matrix);
	//	};
	//	
	//	m_A_cart_nopbc.reserve(8);
	//	m_O_cart_nopbc.reserve(6);
	//	for (const Atom& atom : m_A_direct_pbc) {
	//		m_A_cart_nopbc.emplace_back(atom.m_atom_type, get_cart_pos_nowrap(atom));
	//	}
	//	for (const Atom& atom : m_O_direct_pbc) {
	//		m_O_cart_nopbc.emplace_back(atom.m_atom_type, get_cart_pos_nowrap(atom));
	//	}
	//	m_B_cart_nopbc = Atom(m_B_direct_pbc.m_atom_type, get_cart_pos_nowrap(B));

	//	m_COM_cart_nopbc = convertCoordinates(B.m_position + COM_rel_to_B, cell_matrix);
	//}

	void transformToMinimumImage();
	void transformToDirect();
	void rotateUC() = delete;

};

inline double getSqDistance(const Vector& r, const std::optional<Eigen::Matrix3d>& cell_matrix = std::nullopt) {
	if (!cell_matrix) {
		return r.squaredNorm();
	}
	Eigen::Matrix3d metric { cell_matrix.value().transpose() * cell_matrix.value() };
	return r.dot(metric*r);
}

inline Position getCOM(const Atoms &atoms) {
	Vector COM = Vector::Zero();

	for (const auto &atom : atoms) {
		COM += atom.m_position;
	}

	return COM/atoms.size();
}

inline Eigen::Vector3d convertCoordinates(const Position &pos, const Eigen::Matrix3d &cell_matrix) {
	return cell_matrix*pos;
}

inline double getMinimumImageSqDistance(const Atom& atom1, 
						  const Atom& atom2, 
						  const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {

	Vector dr { atom2.m_position - atom1.m_position };
	dr.array() -= dr.array().round();

	return getSqDistance(dr, cell_matrix);
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

		nearest_neighbors.emplace_back(idx++, getMinimumImageSqDistance(reference_atom, other_atom, cell_matrix));
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
	for (size_t i { 0 }; i < pristine_UC.size(); i++) {
		grad += (local_UC.at(i).m_position - rotatePoint(alpha, pristine_UC.at(i).m_position)).transpose()*(dR*pristine_UC.at(i).m_position);
	}

	return 2*grad;
}

//inline double gradientDescent(double step_size, const UnitCell &pristine_UC, const LocalUC &local_UC) {
//	double alpha { local_UC.m_angle };
//	constexpr double threshold { 1e-12 };
//	constexpr uint max_iter { 1000 };
//
//	for (size_t i : std::ranges::views::iota(max_iter)) {
//		double prev_alpha { alpha };
//		alpha -= step_size * getGradMSD(alpha, pristine_UC.m_A_cart_nopbc, local_UC.m_A_cart_nopbc);
//
//		double diff_alpha { std::abs(prev_alpha - alpha) };
//		if (diff_alpha <= threshold) { 
//			return alpha;
//		}
//	}
//
//	return alpha;
//}
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

//inline void getPolarization(const double step_size, const Atoms &atoms, const std::vector<NNIds> &nearest_neighbors) {
//	// atoms should be same element as nearest_neighbors
//	const helper::UnitCell tetragonal_UC { };
//
//	Atoms local_UC_atoms;
//	local_UC_atoms.reserve(8);
//
//	if (atoms.size() != nearest_neighbors.size()) {
//		throw std::runtime_error("Mismatch in size of atoms and nearest_neighbors!");
//	}
//
//	for (const NNIds &pair_ref_atom_id_nn : nearest_neighbors) { // nearest neighbors for each center atom -> UC
//
//		for (const auto &[nn_id, _]: pair_ref_atom_id_nn) {
//			local_UC_atoms.emplace_back(atoms.at(nn_id));
//		} 
//
//		helper::LocalUC local_UC { helper::LocalUC(local_UC_atoms) };
//		for (Atom &atom : local_UC.m_Sr_atoms) {
//			atom.m_position -= local_UC.m_COM;
//		}
//
//		double alpha { helper::gradientDescent(step_size, tetragonal_UC, local_UC) };
//
//		Vectors displacements;
//		displacements.reserve(8);
//
//		Position rotatedPoint;
//		for (size_t i : std::ranges::views::iota(8)) {
//			rotatedPoint = helper::rotatePoint(alpha, tetragonal_UC.m_A_cart_nopbc.at(i).m_position);
//			displacements.push_back(local_UC.m_A_cart_nopbc.at(i).m_position - rotatedPoint);
//		}
//
//		// calculate BEC
//	}
//}
}
