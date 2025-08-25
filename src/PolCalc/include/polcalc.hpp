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
	HT, HH, APB, Unknown
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
	enum class Rotation {
		right, left, None // std::ranges::rotate right or left
	};
	// A and O contain pairs of opposite atoms (wrt y axis) centered around origin
	std::vector<std::pair<Atom, Atom>> m_A_cart_nopbc { };
	Atom m_B_cart_nopbc { };
	std::vector<std::pair<Atom, Atom>> m_O_cart_nopbc { };
	Position m_COM_cart_nopbc { Position::Zero() };

	private:
	Eigen::Quaterniond m_orientation { 1, 0, 0, 0 };
	std::optional<short> m_permutation_number { std::nullopt };

	void applyRotationO(Atom& atom, double angle, const Vector& axis) {
		// Rodriguez rotation
		Position pos { atom.m_position };
		atom.m_position = pos*cos(angle) + (1-cos(angle))*(pos.dot(axis))*axis + sin(angle)*axis.cross(pos);
	}

	public:
	UnitCell() {};

	UnitCell(AtomType type_A, AtomType type_B, short O_rot_sign /* = +- 1*/, 
		  const Eigen::Quaterniond& orientation, double rot_angle = 3, 
		  AtomType type_O = AtomType::O)
		: m_orientation(orientation)
	{
		m_A_cart_nopbc.reserve(8);

		constexpr double a { 3.905 };
		constexpr double c { a }; 

		Eigen::Vector3d ex = Eigen::Vector3d(1, 0, 0);
		Eigen::Vector3d ey = Eigen::Vector3d(0, 1, 0);
		Eigen::Vector3d ez = Eigen::Vector3d(0, 0, 1);

		auto fill_pristine_A = [&](AtomType type, Position&& pos) {
			Atom first { type, pos };
			pos[1] *= -1;
			Atom second { type, pos }; // mirroring first at xz plane
			m_A_cart_nopbc.emplace_back(std::move(first), std::move(second));
		};

		// front lower left, back lower left, then anticlockwise
		fill_pristine_A(type_A, - 0.5*a*ex + 0.5*a*ey - 0.5*c*ez);
		fill_pristine_A(type_A, 0.5*a*ex + 0.5*a*ey - 0.5*c*ez);
		fill_pristine_A(type_A, 0.5*a*ex + 0.5*a*ey + 0.5*c*ez);
		fill_pristine_A(type_A, -0.5*a*ex + 0.5*a*ey + 0.5*c*ez);

		m_B_cart_nopbc = Atom(type_B, Position::Zero());

		auto fill_pristine_O = [&](AtomType type, Position&& pos_first, Position&& pos_second, const Vector& rot_axis){
			double angle { rot_angle*O_rot_sign };
			Atom first { type, pos_first };
			Atom second { type, pos_second };
			if (rot_axis != Vector(0,0,0)) {
				applyRotationO(first, angle, rot_axis);
				applyRotationO(second, angle, rot_axis);
			}
			m_O_cart_nopbc.emplace_back(std::move(first), std::move(second));
		};

		// filled to resemble the primitive unit cell
		fill_pristine_O(type_O, -0.5*a*ey, 0.5*a*ey, ez); // front, back
		fill_pristine_O(type_O, -0.5*a*ez, 0.5*a*ez, Vector::Zero()); // bottom, top
		fill_pristine_O(type_O, -0.5*a*ex, 0.5*a*ex, ez); // left, right
	}

	struct Displacements {
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> m_A_displacements { }; // same sequence as atoms in UC
		Vector m_B_displacement { };
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> m_O_displacements { };
	};

	Displacements operator-(const UnitCell& other) const {
		auto getDisplacement = [&](const auto& lhs, const auto& rhs, auto& out) {
			for (const auto& [lhs_pair, rhs_pair] : std::ranges::views::zip(lhs, rhs)) {
				Vector first { lhs_pair.first.m_position - rhs_pair.first.m_position };
				Vector second { lhs_pair.second.m_position - rhs_pair.second.m_position };
				out.emplace_back(first, second);
			}
			return out;
		};

		Displacements displacements { };
		displacements.m_A_displacements.reserve(m_A_cart_nopbc.size());
		displacements.m_O_displacements.reserve(m_O_cart_nopbc.size());
		getDisplacement(m_A_cart_nopbc, other.m_A_cart_nopbc, displacements.m_A_displacements);
		getDisplacement(m_O_cart_nopbc, other.m_O_cart_nopbc, displacements.m_O_displacements);
		displacements.m_B_displacement = m_B_cart_nopbc.m_position - other.m_B_cart_nopbc.m_position;

		return displacements;
	}

	template<typename T>
	UnitCell getRotatedUC(const T& R, short permutation_number=0, Rotation permutation_direction=Rotation::None) const {
		UnitCell rotated_UC { *this };
		Eigen::Quaterniond unit_quaternion { R }; // if R is rotation matrix, convert to unit quaternion

		auto rotate_points = [&](std::vector<std::pair<Atom, Atom>>& pairs) {
			std::vector<std::pair<Atom, Atom>> rotated_atoms;
			for (auto& pair : pairs) {
				Atom first { pair.first.m_atom_type, unit_quaternion*pair.first.m_position };
				Atom second { pair.second.m_atom_type, unit_quaternion*pair.second.m_position };
				rotated_atoms.emplace_back(first, second);
			}
			return rotated_atoms;
		};

		auto permute = [&rotated_UC, &permutation_number, &permutation_direction](std::vector<std::pair<Atom, Atom>>& pairs) {
			if (permutation_direction == Rotation::None || permutation_number == 0) {
				return;
			}
			if (permutation_direction == Rotation::left) {
				std::ranges::rotate(pairs, pairs.begin()+permutation_number);
			}
			else {
				std::ranges::rotate(pairs, pairs.end()-permutation_number);
			}
			rotated_UC.m_permutation_number = permutation_number;
		};

		rotated_UC.m_A_cart_nopbc = rotate_points(rotated_UC.m_A_cart_nopbc);
		rotated_UC.m_O_cart_nopbc = rotate_points(rotated_UC.m_O_cart_nopbc);
		rotated_UC.m_orientation *= unit_quaternion;

		permute(rotated_UC.m_A_cart_nopbc);

		return rotated_UC;
	}

	Eigen::Quaterniond getInitialOrientation() const {
		return m_orientation;
	}

	std::expected<short, std::string> getInitialPermutation() const {
		if (!m_permutation_number) {
			return std::unexpected("permutation number not set");
		}
		return m_permutation_number.value();
	}
};

class LocalUC : UnitCell {

public:
	enum class DWSide {
		left, right, center, Unknown
	};

	std::vector<std::pair<Atom, Atom>> m_A_direct_pbc { };
	Atom m_B_direct_pbc { };
	std::vector<std::pair<Atom, Atom>> m_O_direct_pbc { };
	std::optional<std::tuple<Eigen::Quaterniond, short, Rotation>> m_orientation; // recipe how to transform unrotated reference cell: Quaternion for the spacial rotation, Rotation for rotating the Sr edges (permutation) either right or left by one
	std::optional<short> O_rot_sign;
	DWSide m_side { DWSide::Unknown };
	DWType m_type { DWType::Unknown };

	using UnitCell::m_A_cart_nopbc;
	using UnitCell::m_B_cart_nopbc;
	using UnitCell::m_O_cart_nopbc;
	using UnitCell::m_COM_cart_nopbc;
	using UnitCell::operator-;
	using UnitCell::Rotation;

private:
	static inline std::optional<std::tuple<Eigen::Quaterniond, short, Rotation>> m_right_init_orientation;
	static inline std::optional<std::tuple<Eigen::Quaterniond, short, Rotation>> m_left_init_orientation;
	Eigen::Matrix3d m_metric;

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
			m_side = DWSide::center;
		}
		else if (ref.x() > DW_center_x) {
			m_side = DWSide::right;
		}
		else {
			m_side = DWSide::left;
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
		std::vector<std::tuple<Atom, Atom, Vector>> matches;
		size_t matches_size { atoms.size() / 2 };
		matches.reserve(matches_size);
		std::pair<int, int> pair_along_y;

		for (size_t i { 0 }; i < atoms.size(); i++) {
			for (size_t j { 0 }; j < i; j++) {
				Position pos1 { atoms[i].m_position };
				Position pos2 { atoms[j].m_position };
				
				double cos { pos1.dot(m_metric*pos2) / sqrt(pos1.dot(m_metric*pos1) * pos2.dot(m_metric*pos2)) };
				if (1+cos < cos_tolerance) {
					matches.emplace_back(atoms[i], atoms[j], pos1-pos2);
				}
			}
		}
		if (matches.size() != matches_size) {
			throw std::runtime_error("Matches are " + std::to_string(matches.size()) + " but should be " + std::to_string(matches_size));
		}
		// find pair along y axis and put it at idx 0 of matches_sorted, rest is random order
		std::vector<std::pair<Atom, Atom>> matches_sorted;
		matches_sorted.reserve(matches_size);
		std::pair<int, double> closest_y { };
		Vector ey { 0, 1, 0 };

		for (size_t i { }; i < matches_size; i++) {
			Vector vec { std::get<2>(matches.at(i)) };
			double cur_cos { std::abs(vec.dot(m_metric*ey) / sqrt(vec.dot(m_metric*vec)*ey.dot(m_metric*ey))) };
			if (cur_cos > closest_y.second) { 
				closest_y.first = i;
				closest_y.second = cur_cos;
			}
		}

		auto get_match_entry = [&](int idx) { 
			switch (idx) {
				case 0:
					return std::get<0>(matches.at(closest_y.first));
				case 1:
					return std::get<1>(matches.at(closest_y.first));
				default:
					std::unreachable();
			}
		};

		// put the front O with a negative y value at first, the one with the positive y value second
		// similar to UnitCell
		if (get_match_entry(0).m_position.y() < get_match_entry(1).m_position.y()) {
			matches_sorted.emplace_back(get_match_entry(0), get_match_entry(1));
		}
		else {
			matches_sorted.emplace_back(get_match_entry(1), get_match_entry(0));
		}
		matches.erase(matches.begin()+closest_y.first);
		for (auto& match : matches) {
			matches_sorted.emplace_back(std::get<0>(match), std::get<1>(match));
		}
		return matches_sorted;
	}

public:
	std::expected<std::tuple<Eigen::Quaterniond, short, Rotation>, std::string> getInitialOrientation() const {
		if (!m_left_init_orientation || !m_right_init_orientation) {
			return std::unexpected("Initial rotation not set!");
		}
		else if (m_side == DWSide::left || m_side == DWSide::center) {
			return m_left_init_orientation.value();
		}
		else if (m_side == DWSide::right) {
			return m_right_init_orientation.value();
		}
		else {
			return std::unexpected("DW side unknown, initilization failed");
		}
	}

	std::expected<std::tuple<Eigen::Quaterniond, short, Rotation>, std::string> getOrientation() const {
		if (m_orientation) {
			return m_orientation.value();
		}
		return std::unexpected("Rotation not set!");
	}

	explicit LocalUC(const Atoms& A, const Atom& B, const Atoms& O, DWType DW_type, const Eigen::Matrix3d& cell_matrix, double DW_center_x = 0.5, double tolerance = 1e-3) 
		: m_B_direct_pbc(B), m_metric(cell_matrix.transpose() * cell_matrix), m_type(DW_type)
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

		for (const Atom& corner : A_rel_to_B) {
			Atom corner_rel_to_COM { corner.m_atom_type, minimumImage(corner.m_position - COM_rel_to_B) };
			double angle { getAngle(cell_matrix*corner_rel_to_COM.m_position) };
			(corner_rel_to_COM.m_position[1] < 0 ? atoms_lower : atoms_upper).emplace_back(corner_rel_to_COM, angle);
		}

		// first take the corner with the largest x value (add it at idx 0), get angles of remaining corners and split corners into upper and lower
		std::pair<Atom, double>& rightmost_corner_upper { atoms_upper.at(0) };
		// find corner
		for (const auto& corner : atoms_upper) {
			if (corner.first.m_position.x() > rightmost_corner_upper.first.m_position.x()) {
				rightmost_corner_upper = corner;
			}
		}
		auto result_upper = std::ranges::find_if(atoms_upper, [&](const std::pair<Atom, double>& pair) { 
			return pair.first.m_position.isApprox(rightmost_corner_upper.first.m_position);
		});
		// move hit to first position
		std::ranges::swap(*result_upper, atoms_upper.front());

		//now find the matching Sr in the lower plane
		std::pair<Atom, double>& rightmost_corner_lower { atoms_lower.at(0) };
		for (const auto& corner : atoms_lower) {
			double dist { (corner.first.m_position - rightmost_corner_upper.first.m_position).squaredNorm() };
			if (dist < (corner.first.m_position - rightmost_corner_lower.first.m_position).squaredNorm()) {
				rightmost_corner_lower = corner;
			}
		}
		auto result_lower = std::ranges::find_if(atoms_lower, [&](const std::pair<Atom, double>& pair) { 
			return pair.first.m_position.isApprox(rightmost_corner_lower.first.m_position);
		});
		std::ranges::swap(*result_lower, atoms_lower.front());

		rebalancePlanes(atoms_upper, atoms_lower);

		// wrap from direct atoms COM centered back to direct coordinates with pbc
		wrapDirectCoordinates(atoms_upper, B.m_position + COM_rel_to_B);
		wrapDirectCoordinates(atoms_lower, B.m_position + COM_rel_to_B);

		// sort from idx 1 onwarts
		auto sort_angles = [](auto& arr) { 
			std::ranges::sort(arr.begin()+1, arr.end(), [](auto& pair1, auto& pair2) { 
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

	std::expected<void, std::string> setInitialOrientation(const std::tuple<Eigen::Quaterniond, short, Rotation>& unit_quaternion) { 
		if (m_side == DWSide::center) {
			return std::unexpected("Cell in DW center!"); 
		}

		if (m_left_init_orientation && m_right_init_orientation) {
			return std::unexpected("Initial Orientation already set!");
		}

		const Eigen::Quaterniond& quaternion { std::get<0>(unit_quaternion) };
		const short permutation_number { std::get<1>(unit_quaternion) };

		Eigen::Quaterniond rot_90_y { cos(M_PI/4), 0, sin(M_PI/4), 0 };
		const double angle { [&]() {
			Vector z_rotated { quaternion*Vector(0,0,1) };
			double angle { atan2(z_rotated.z(), z_rotated.x()) };
			return angle < 0 ? angle+2*M_PI : angle; // get the angle of rotated z axis around y axis
		}() };

		// for APBs take into account that z axis remains unchanged, this function is wrong then
		bool quadrant_2 { M_PI/2 < angle && angle < M_PI };
		bool quadrant_4 { 3*M_PI/2 < angle && angle < 2*M_PI };

		auto get_permutation_number = [](short n) -> short {
			short m { static_cast<short>(n%4) };
			if (m >= 0) {
				return m;
			}
			return m+4;
		};

		if (m_type == DWType::HT) {
			if (m_side == DWSide::left) {
				m_left_init_orientation = unit_quaternion;
				if (quadrant_2 || quadrant_4) {
					m_right_init_orientation = std::make_tuple(rot_90_y*quaternion, get_permutation_number(permutation_number+1), Rotation::left);
				}
				else {
					m_right_init_orientation = std::make_tuple(rot_90_y.conjugate()*quaternion, get_permutation_number(permutation_number-1), Rotation::right);
				}
			}
			else {
				m_right_init_orientation = unit_quaternion;
				if (quadrant_2 || quadrant_4) {
					m_left_init_orientation = std::make_tuple(rot_90_y*quaternion, get_permutation_number(permutation_number+1), Rotation::left);
				}
				else {
					m_left_init_orientation = std::make_tuple(rot_90_y.conjugate()*quaternion, get_permutation_number(permutation_number-1), Rotation::right);
				}
			}
		}
		else {
			if (m_side == DWSide::left) {
				m_left_init_orientation = unit_quaternion;
				if (quadrant_2 || quadrant_4) {
					m_right_init_orientation = std::make_tuple(rot_90_y.conjugate()*quaternion, get_permutation_number(permutation_number-1), Rotation::right);
				}
				else {
					m_right_init_orientation = std::make_tuple(rot_90_y*quaternion, get_permutation_number(permutation_number+1), Rotation::left);
				}
			}
			else {
				m_right_init_orientation = unit_quaternion;
				if (quadrant_2 || quadrant_4) {
					m_left_init_orientation = std::make_tuple(rot_90_y.conjugate()*quaternion, get_permutation_number(permutation_number-1), Rotation::right);
				}
				else {
					m_left_init_orientation = std::make_tuple(rot_90_y*quaternion, get_permutation_number(permutation_number+1), Rotation::left);
				}
			}
		}
		return {};
	}

	void sortOs(const Eigen::Quaterniond& unit_quaternion) {
		// only sorts cartesian coordinates
		Vector ex { 1, 0, 0 };
		Vector ez { 0, 0, 1 };
		// rotate ex
		ex = unit_quaternion*ex;
		ez = unit_quaternion*ez;
		
		std::pair<size_t, double> projection_z { };
		for (size_t i { 1 }; i < m_O_cart_nopbc.size(); i++) {
			Vector pair_vector { m_O_cart_nopbc.at(i).first.m_position - m_O_cart_nopbc.at(i).second.m_position };
			double projection { std::abs(pair_vector.dot(ez)) };
			if (projection > projection_z.second) {
				projection_z.first = i;
				projection_z.second = projection;
			}
		}
		// atom with the higher value is at that position [1] bottom, top [2] left, right
		std::ranges::swap(m_O_cart_nopbc.at(1), m_O_cart_nopbc.at(projection_z.first));
		// make sure that first and second positions are the same as in unit cell
		auto swap_elements = [&](std::pair<Atom, Atom>& pair, const Vector& axis) {
			Position pos1 { pair.first.m_position - m_B_cart_nopbc.m_position };

			if (pos1.dot(axis) > 0) {
				Atom temp;	
				temp = pair.first;
				pair.first = pair.second;
				pair.second = temp;
			}
		};
		swap_elements(m_O_cart_nopbc.at(1), ez); // sort to bottom top
		swap_elements(m_O_cart_nopbc.at(2), ex); // sort to left right
	}

	LocalUC getCenteredUC() const {
		LocalUC centered_uc { *this };
		auto center = [&](std::vector<std::pair<Atom, Atom>>& pairs) {
			for (std::pair<Atom, Atom>& pair : pairs) {
				pair.first.m_position -= m_COM_cart_nopbc;
				pair.second.m_position -= m_COM_cart_nopbc;
			}
		};

		center(centered_uc.m_A_cart_nopbc);
		center(centered_uc.m_O_cart_nopbc);
		centered_uc.m_B_cart_nopbc.m_position -= m_COM_cart_nopbc;
		return centered_uc;
	}
	
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

inline Eigen::Matrix3d getRotationMatrix(const Eigen::Quaterniond& unit_quaternion) {
	double q0 = unit_quaternion.w();
	double q1 = unit_quaternion.x();
	double q2 = unit_quaternion.y();
	double q3 = unit_quaternion.z();
	 
	double sq_q0 = pow(q0, 2);
	double sq_q1 = pow(q1, 2);
	double sq_q2 = pow(q2, 2);
	double sq_q3 = pow(q3, 2);

	Eigen::Matrix3d rotation_matrix;
	rotation_matrix << sq_q0+sq_q1-0.5,  q1*q2-q0*q3,    q1*q3+q0*q2,
					     q1*q2+q0*q3,  sq_q0+sq_q2-0.5,  q2*q3-q0*q1,
					     q1*q3-q0*q2,    q2*q3+q0*q1,  sq_q0+sq_q3-0.5;

	return 2*rotation_matrix;
}

inline Eigen::Quaterniond gradientDescent(const UnitCell& pristine_UC, const LocalUC &local_UC, double step_size, const std::tuple<Eigen::Quaterniond, short, UnitCell::Rotation>& init_orientation) {
	constexpr double threshold { 1e-12 };
	constexpr size_t max_iter { 1000 };

	auto sq_dist = [](UnitCell::Displacements displacements) {
		double sq_dist { };
		for (const auto& pair : displacements.m_A_displacements) {
			sq_dist += pair.first.squaredNorm();
			sq_dist += pair.second.squaredNorm();
		}
		return sq_dist;
	};

	Eigen::Quaterniond current_unit_quaternion { std::get<0>(init_orientation).normalized() };
	UnitCell pristine_UC_init { pristine_UC.getRotatedUC(current_unit_quaternion, std::get<1>(init_orientation), std::get<2>(init_orientation))};
	LocalUC local_UC_centered { local_UC.getCenteredUC() };

	size_t cur_iter { };
	double cur_sq_dist { sq_dist(local_UC - pristine_UC_init )};
	double diff_A { std::numeric_limits<double>::infinity() };



	auto gradR = [&current_unit_quaternion](size_t i){
		Eigen::Matrix3d grad_R;

		const double& q0 { current_unit_quaternion.w() };
		const double& q1 { current_unit_quaternion.x() };
		const double& q2 { current_unit_quaternion.y() };
		const double& q3 { current_unit_quaternion.z() };

		switch (i) {
			// order q1,q2,q3,q0 because Eigen stores the coefficient this way
			case 0:
				grad_R << 2*q1, q2,  q3,
						   q2,   0, -q0,
						   q3,  q0,   0;
				return 2*grad_R;
			case 1:
				grad_R <<  0,  q1,  q0,
						  q1, 2*q2, q3,
						 -q0,  q3,   0;
				return 2*grad_R;
			case 2:
				grad_R <<  0, -q0,  q1,
						  q0,   0,  q2,
						  q1,  q2, 2*q3;
				return 2*grad_R;
			case 3:
				grad_R << 2*q0, -q3,   q2,
						   q3,  2*q0, -q1,
						  -q2,   q1,  2*q0;
				return 2*grad_R;
			default:
				std::unreachable();
		}
	};

	while (cur_iter++ < max_iter && diff_A > threshold) {
		const auto& current_quaternion_coeff { current_unit_quaternion.coeffs() };
		Eigen::Quaterniond new_unit_quaternion { 1, 0, 0, 0 };
		auto& new_quaternion_coeff { new_unit_quaternion.coeffs() };
		for (size_t j { }; j < 4; j++) {
			const Eigen::Matrix3d grad_R { gradR(j) };
			double sum { };
			for (size_t i { }; i < 4; i++) {
				const std::pair<Atom, Atom>& pristine_atoms_A = pristine_UC_init.m_A_cart_nopbc[i];
				const std::pair<Atom, Atom>& local_atoms_A = local_UC.m_A_cart_nopbc[i];

				sum += local_atoms_A.first.m_position.dot(grad_R*pristine_atoms_A.first.m_position);
				sum += local_atoms_A.second.m_position.dot(grad_R*pristine_atoms_A.second.m_position);
			}

			new_quaternion_coeff[j] = current_quaternion_coeff[j] + 2*step_size*sum;
		}

		current_unit_quaternion = new_unit_quaternion.normalized();
		Eigen::Matrix3d rotation_matrix { getRotationMatrix(current_unit_quaternion) };

		
		double new_sq_dist { sq_dist(local_UC - pristine_UC_init.getRotatedUC(rotation_matrix)) };

		diff_A = std::abs(new_sq_dist - cur_sq_dist);
		cur_sq_dist = new_sq_dist;
	}

	return current_unit_quaternion;
}

inline void findInitialOrientation(LocalUC& local_UC, double step_size) {
	Eigen::Matrix3d rotation_matrix_90;
	rotation_matrix_90 << 0, 0, 1,
						  0, 1, 0,
						 -1, 0, 0;

	UnitCell unit_cell { AtomType::Sr, AtomType::Ti, +1, { 1, 0, 0, 0 } };

	auto center = [&local_UC](std::vector<std::pair<Atom, Atom>>& atom_pairs) {
		for (auto& pair : atom_pairs) {
			pair.first.m_position -= local_UC.m_COM_cart_nopbc;
			pair.second.m_position -= local_UC.m_COM_cart_nopbc;
		}
	};
	// only Sr and O centered
	const LocalUC pseudo_unit_cell_centered { [&]() {
		LocalUC A_O_centered_cell { local_UC };
		center(A_O_centered_cell.m_A_cart_nopbc);
		center(A_O_centered_cell.m_O_cart_nopbc);
		return A_O_centered_cell;
	}() };

	std::tuple<Eigen::Quaterniond, short, UnitCell::Rotation> initial_orientation { { 1, 0, 0, 0 }, 0, UnitCell::Rotation::None };
	Eigen::Quaterniond unit_quaternion = gradientDescent(unit_cell, local_UC, step_size, initial_orientation); // start from unrotated uc 
	std::pair<double, UnitCell> best_uc { std::numeric_limits<double>::infinity(), UnitCell() }; // dist front O and corresponding uc
	

	short permutation_number {};
	for (std::size_t i { }; i<4; i++) {
		if (i>0) {
			unit_cell = unit_cell.getRotatedUC(rotation_matrix_90, ++permutation_number, UnitCell::Rotation::left);
		}

		const Position& pristine_uc_top_O { (unit_cell.getRotatedUC(unit_quaternion)).m_O_cart_nopbc.at(0).first.m_position };
		const Position& pseudo_unit_cell_uc_top_O { pseudo_unit_cell_centered.m_O_cart_nopbc.at(0).first.m_position };
		double cur_top_O_sq_distance { (pristine_uc_top_O - pseudo_unit_cell_uc_top_O).squaredNorm() };
		// update best distance
		if (cur_top_O_sq_distance < best_uc.first) {
			best_uc.first = cur_top_O_sq_distance;
			best_uc.second = unit_cell;
		}
	}
	// calculate quaternion to find rotated coordinate system
	Eigen::Quaterniond initial_quaternion { best_uc.second.getInitialOrientation() };
	Eigen::Quaterniond best_orientation { initial_quaternion*unit_quaternion };
	
	auto final_orientation	{ 
		best_uc.second.getInitialPermutation()
			.transform([&](short value) {
				return std::make_tuple(best_orientation, value, UnitCell::Rotation::left);
			})
	};
	
	if (!final_orientation) { // temporary, implement proper error handling, save progress, then terminate program
		throw std::runtime_error(std::string("In getInitialPermutation: " + final_orientation.error()));
	}

	std::expected<void, std::string> res = local_UC.setInitialOrientation(final_orientation.value());

	if (!res) {
		throw std::runtime_error("set initial Orientation failed");
	}

	local_UC.m_orientation = final_orientation.value();
}

inline void getOP(const LocalUC& local_UC, const UnitCell& pristine_UC) {
}

inline void getPolarization(const LocalUC& local_UC) { 
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
    for (uint i = 0; i < head && std::getline(file, line); ++i) { /* skip */ }

    Positions positions;
    long line_num = static_cast<long>(head);

    while (std::getline(file, line)) {
        ++line_num;
        if (tail_start > 0 && line_num >= tail_start) break;

        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) {
            continue;
        }
        positions.emplace_back(x, y, z);
    }

    return positions;
}

inline std::expected<AtomPositions, std::string> sortPositionsByType(const Positions& positions, 
																	 const size_t N_Sr, 
																	 const size_t N_Ti, 
																	 const size_t N_O) {
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

		auto res = helper::findNearestN(ref_atom, atoms, n, exclude_idx, cell_matrix, sort) 
			.transform([&nearest_neighbors](auto&& res) { nearest_neighbors.push_back(std::move(res)); });

		if (!res) {
			return std::unexpected("On iteration " + std::to_string(atom_id) + ", in findNearestN: " + res.error());
		}

		atom_id++;
	}

	return nearest_neighbors;
}

inline std::vector<helper::LocalUC> createLocalUCs(const Atoms& A, const Atoms& B, const Atoms& O,
						   const std::vector<NNIds>& A_NNIds_all, const std::vector<NNIds>& O_NNIds_all,
						   DWType DW_type, Eigen::Matrix3d& cell_matrix) {

	std::vector<helper::LocalUC> local_UCs;
	local_UCs.reserve(B.size());

	const auto B_ids = std::ranges::views::iota(size_t {0}, B.size()) ;
	for (const auto& [A_NNIds, B_id, O_NNIds] : std::ranges::views::zip(A_NNIds_all, B_ids, O_NNIds_all)) {

		auto fill_atoms = [](const auto& NNids, const Atoms& atoms, Atoms& container) {
			for (auto NN : NNids) {
				container.emplace_back(atoms.at(NN.first));
			}
		};

		Atoms A_local; A_local.reserve(A_NNIds.size());
		Atom B_local { B.at(B_id) };
		Atoms O_local; O_local.reserve(O_NNIds.size());
		
		fill_atoms(A_NNIds, A, A_local);
		fill_atoms(O_NNIds, O, O_local);

		local_UCs.emplace_back(std::move(A_local), std::move(B_local), std::move(O_local), DW_type, cell_matrix);
	}
	return local_UCs;
}

inline void getOPAndPolarization(std::vector<helper::LocalUC>& local_UCs, double step_size) {
	// write a custom find/set initial orientation function for APBs 
	std::vector<helper:: LocalUC> DW_center_init; // containing all center DWs picked befor local z axis could be determined
	
	helper::UnitCell pristine_UC_sp { AtomType::Sr, AtomType::Ti, 1, { 1, 0, 0, 0 } }; // sigma +1 UC
	helper::UnitCell pristine_UC_sn { AtomType::Sr, AtomType::Ti, -1, { 1, 0, 0, 0 } }; // sigma -1 UC

	bool is_first { true };
	for (auto&& local_UC : local_UCs) {
		if (is_first && local_UC.m_side == helper::LocalUC::DWSide::center) { // avoid artifacts due to different O placement in DW center cells
			DW_center_init.emplace_back(local_UC);
			continue;
		}

		if (is_first) {
			helper::findInitialOrientation(local_UC, step_size);
			is_first = false;
		}
		else {
			auto local_UC_initial = local_UC.getInitialOrientation().value();
			Eigen::Quaterniond best_quaternion { helper::gradientDescent(pristine_UC_sp, local_UC,  0.05, local_UC_initial) };
			local_UC.m_orientation = std::make_tuple(best_quaternion, std::get<1>(local_UC_initial), std::get<2>(local_UC_initial));

			Position front_O_sp { pristine_UC_sp.getRotatedUC(best_quaternion).m_O_cart_nopbc.at(0).first.m_position };
			Position front_O_sn { pristine_UC_sn.getRotatedUC(best_quaternion).m_O_cart_nopbc.at(0).first.m_position };
			Position local_UC_front_O { local_UC.m_O_cart_nopbc.at(0).first.m_position };

			double displacement_sp { (front_O_sp - local_UC_front_O).squaredNorm() };
			double displacement_sn { (front_O_sn - local_UC_front_O).squaredNorm() };

			displacement_sp < displacement_sn ? local_UC.O_rot_sign = 1 : local_UC.O_rot_sign = -1;

			local_UC.sortOs(best_quaternion);

			// calculate OP and Polarization
		};
	}
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
