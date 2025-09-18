#pragma once

#include <iomanip>
#include <iostream>
#include <fstream>
#include <deque>
#include <stdexcept>
#include <expected>
#include <print>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <ranges>
#include <utility>
#include <optional>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numeric>

// everything with right handed coordinate system looking along +y

// TODO use openACC to parallelise on cpu or cuda
// for multiple frames reset the static orientation in the LocalUCs
// write different behavior for other filetypes (CONTCAR, XDATCAR, xyz, ...), calculate atom numbers??? 
namespace PolCalc {

using Position = Eigen::Vector3d;
using Positions = std::vector<Position>;
using Vector = Eigen::Vector3d;
using Vectors = std::vector<Vector>;
using NNIds = std::vector<std::pair<size_t, double>>; 

enum class DWType {
	HT=0, HH, APB, Unknown
};

enum class AtomType {
	Sr=0, Ti, O, Unknown
};

enum class Observable { 
	OP, Polarization
};

constexpr std::array<double, 3> ATOM_MASSES_U { 87.62, 47.867, 15.999 }; // Sr, Ti, O; use a map when introducing more types

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

	Atoms m_A {};
	Atoms m_B {};
	Atoms m_O {};

	AtomPositions() = default;
	AtomPositions(const size_t N_Sr, const size_t N_Ti, const size_t N_O) {
		m_A.reserve(N_Sr);
		m_B.reserve(N_Ti);
		m_O.reserve(N_O);
	}
};

struct ObservableData {
	Observable m_observable;
	std::vector<double> m_bin_center;
	Vectors m_observable_average;
	Vectors m_observable_variance;

	ObservableData() = delete;

	ObservableData(size_t arr_size) {
		m_bin_center.reserve(arr_size);
		m_observable_average.reserve(arr_size);
		m_observable_variance.reserve(arr_size);
	}

	ObservableData(std::vector<double> center, Vectors average, Vectors variance, Observable observable) 
	: m_bin_center(center), m_observable_average(average), m_observable_variance(variance), m_observable(observable)
	{} 
};

std::expected<std::vector<NNIds>, std::string> getNearestNeighbors(const Atoms& atoms, 
																   const Atoms& reference_atoms,
																   const size_t n, 
																   const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt,
																   const bool wrap = true);


namespace helper {

inline Eigen::Vector3d convertCoordinates(const Position &pos, const Eigen::Matrix3d &cell_matrix);

static inline Position getCOM(const Atoms &atom);

class UnitCell {
	// tetragonal unit cell with COM at origin
public:
	enum class Rotation {
		None=0, right, left  // std::ranges::rotate right or left
	};
	// A and O contain pairs of opposite atoms (wrt y axis) centered around origin
	std::vector<std::pair<Atom, Atom>> m_A_cart_nopbc { };
	Atom m_B_cart_nopbc { };
	std::vector<std::pair<Atom, Atom>> m_O_cart_nopbc { };
	Position m_COM_cart_nopbc { Position::Zero() };
	double m_cell_volume {};

private:
	Eigen::Quaterniond m_orientation { 1, 0, 0, 0 };
	std::pair<short, Rotation> m_permutation_number { };

	void applyRotationO(Atom& atom, double angle, const Vector& axis) {
		// Rodriguez rotation
		Position pos { atom.m_position };
		atom.m_position = pos*cos(angle) + (1-cos(angle))*(pos.dot(axis))*axis + sin(angle)*axis.cross(pos);
	}

public:
	UnitCell() {
		m_A_cart_nopbc.reserve(4);
		m_O_cart_nopbc.reserve(3);
	}

	UnitCell(AtomType type_A, AtomType type_B, double lattice_const = 3.905) {
		m_A_cart_nopbc.reserve(4);
		m_O_cart_nopbc.reserve(3);

		const double a { lattice_const };
		const double c { a }; 

		m_cell_volume = a*a*c;

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
		fill_pristine_A(type_A, - 0.5*a*ex - 0.5*a*ey - 0.5*c*ez);
		fill_pristine_A(type_A, 0.5*a*ex - 0.5*a*ey - 0.5*c*ez);
		fill_pristine_A(type_A, 0.5*a*ex - 0.5*a*ey + 0.5*c*ez);
		fill_pristine_A(type_A, -0.5*a*ex - 0.5*a*ey + 0.5*c*ez);

		m_B_cart_nopbc = Atom(type_B, Position::Zero());

		auto fill_pristine_O = [&](AtomType type, Position&& pos_first, Position&& pos_second){
			Atom first { type, pos_first };
			Atom second { type, pos_second };
			m_O_cart_nopbc.emplace_back(std::move(first), std::move(second));
		};

		fill_pristine_O(AtomType::O, -0.5*a*ey, 0.5*a*ey); // front, back
		fill_pristine_O(AtomType::O, -0.5*a*ez, 0.5*a*ez); // bottom, top
		fill_pristine_O(AtomType::O, -0.5*a*ex, 0.5*a*ex); // left, right
	}

	UnitCell(AtomType type_A, AtomType type_B, short O_rot_sign /* = +- 1*/, 
		  const Eigen::Quaterniond& orientation, double rot_angle = 3*M_PI/180, 
		  double lattice_const = 3.905, AtomType type_O = AtomType::O)
	: m_orientation(orientation)
	{
		m_A_cart_nopbc.reserve(4);
		m_O_cart_nopbc.reserve(3);

		const double a { lattice_const };
		const double c { a }; 
		m_cell_volume = a*a*c;

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
		fill_pristine_A(type_A, - 0.5*a*ex - 0.5*a*ey - 0.5*c*ez);
		fill_pristine_A(type_A, 0.5*a*ex - 0.5*a*ey - 0.5*c*ez);
		fill_pristine_A(type_A, 0.5*a*ex - 0.5*a*ey + 0.5*c*ez);
		fill_pristine_A(type_A, -0.5*a*ex - 0.5*a*ey + 0.5*c*ez);

		m_B_cart_nopbc = Atom(type_B, Position::Zero());

		auto fill_pristine_O = [&](AtomType type, Position&& pos_first, Position&& pos_second, const Vector& rot_axis) {
			Atom first { type, pos_first };
			Atom second { type, pos_second };
			if (rot_angle == 0.0) {
				m_O_cart_nopbc.emplace_back(std::move(first), std::move(second));
				return;
			}

			double angle { rot_angle*O_rot_sign };
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
	// add expected value later
	void RotateUC(const T& R, short permutation_number=0, Rotation permutation_direction=Rotation::None) {
		if (permutation_number > 3 || permutation_number < 0) {
			throw std::runtime_error("permutation_number has to be in [0,3]!");
		}

		Eigen::Quaterniond unit_quaternion { R }; // if R is rotation matrix, convert to unit quaternion
		unit_quaternion.normalize();

		auto rotate_points = [&](std::vector<std::pair<Atom, Atom>>& pairs) {
			std::vector<std::pair<Atom, Atom>> rotated_atoms;
			for (auto& pair : pairs) {
				Atom first { pair.first.m_atom_type, unit_quaternion*pair.first.m_position };
				Atom second { pair.second.m_atom_type, unit_quaternion*pair.second.m_position };
				rotated_atoms.emplace_back(first, second);
			}
			return rotated_atoms;
		};

		// for now only cartesian coordinates are changed, direct are kept as is
		auto permute = [this, permutation_number, permutation_direction](std::vector<std::pair<Atom, Atom>>& pairs) {
			if (permutation_direction == Rotation::None || permutation_number == 0) {
				return;
			}
			if (permutation_direction == Rotation::left) {
				std::ranges::rotate(pairs, pairs.begin()+permutation_number);
			}
			else {
				std::ranges::rotate(pairs, pairs.end()-permutation_number);
			}
			m_permutation_number = std::make_pair(permutation_number, permutation_direction);
		};

		m_A_cart_nopbc = rotate_points(m_A_cart_nopbc);
		m_O_cart_nopbc = rotate_points(m_O_cart_nopbc);
		m_orientation *= unit_quaternion;

		permute(m_A_cart_nopbc);
	}

	template<typename T>
	UnitCell getRotatedUC(const T& R, short permutation_number=0, Rotation permutation_direction=Rotation::None) const {
		UnitCell rotated_UC { *this };
		rotated_UC.RotateUC(R, permutation_number, permutation_direction);
		return rotated_UC;
	}

	Eigen::Quaterniond getInitialOrientation() const {
		return m_orientation;
	}

	std::pair<short, Rotation> getInitialPermutation() const {
		return m_permutation_number;
	}
};

class LocalUC : UnitCell {

public:
	enum class DWSide {
		left, right, center, Unknown
	};

	enum class PhaseFactor {
		negative = -1, positive = 1
	};

	std::vector<std::pair<Atom, Atom>> m_A_direct_pbc { };
	Atom m_B_direct_pbc { };
	std::vector<std::pair<Atom, Atom>> m_O_direct_pbc { };

	std::optional<std::tuple<Eigen::Quaterniond, short, Rotation>> m_orientation; // recipe how to transform unrotated reference cell: Quaternion for the spacial rotation, Rotation for rotating the Sr edges (permutation) either right or left by one
	std::optional<short> m_O_rot_sign;

	DWSide m_side { DWSide::Unknown };
	DWType m_type { DWType::Unknown };

	using UnitCell::m_A_cart_nopbc;
	using UnitCell::m_B_cart_nopbc;
	using UnitCell::m_O_cart_nopbc;
	using UnitCell::m_COM_cart_nopbc;

	double m_local_lattice_constant;
	std::optional<Vector> m_local_OP_local_frame;
	std::optional<Vector> m_local_OP_global_frame;
	std::optional<Vector> m_local_polarization_local_frame;
	std::optional<Vector> m_local_polarization_global_frame;
	using UnitCell::operator-;
	using UnitCell::Rotation;

	void rotateUC() = delete;

private:
	static inline std::optional<std::tuple<Eigen::Quaterniond, short, Rotation>> m_right_init_orientation;
	static inline std::optional<std::tuple<Eigen::Quaterniond, short, Rotation>> m_left_init_orientation;
	static inline std::optional<double> m_global_lattice_constant;
	std::optional<PhaseFactor> m_phase_factor;
	Eigen::Matrix3d m_metric;


	static Position minimumImage(const Position& pos) {
		return (pos.array() - (pos.array() + 0.5 - 1e-11).floor()).matrix();
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
		// find pair along y axis and put it at idx 0 of matches_sorted, put pair along 01 direction at [1] ...
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

	std::pair<std::vector<std::pair<Atom, double>>, std::vector<std::pair<Atom, double>>> buildAPBCell(const Atoms& A_rel_to_B, const Atom& B, const Position& COM_rel_to_B, const Eigen::Matrix3d& cell_matrix) {
		std::vector<std::pair<Atom, double>> atoms_upper, atoms_lower;
		atoms_upper.reserve(4);
		atoms_lower.reserve(4);

		for (const Atom& corner : A_rel_to_B) {
			Atom corner_rel_to_COM { corner.m_atom_type, minimumImage(corner.m_position - COM_rel_to_B) };
			double angle { getAngle(cell_matrix*corner_rel_to_COM.m_position) };
			(corner_rel_to_COM.m_position[1] < 0 ? atoms_lower : atoms_upper).emplace_back(corner_rel_to_COM, angle);
		}

		rebalancePlanes(atoms_upper, atoms_lower);

		wrapDirectCoordinates(atoms_upper, B.m_position + COM_rel_to_B);
		wrapDirectCoordinates(atoms_lower, B.m_position + COM_rel_to_B);

		auto label_half = [](std::vector<std::pair<Atom, double>>& arr) {
			// label the corners the same way as in pristine UC (no perm)
			std::ranges::sort(arr, [](auto& pair1, auto& pair2) { 
				return pair1.second < pair2.second;
			});
			std::ranges::rotate(arr, arr.begin()+2);
		};

		label_half(atoms_upper);
		label_half(atoms_lower);

		return std::make_pair(atoms_upper, atoms_lower);
	}

	std::pair<std::vector<std::pair<Atom, double>>, std::vector<std::pair<Atom, double>>> buildTwinCell(const Atoms& A_rel_to_B, const Atom& B, const Position& COM_rel_to_B, const Eigen::Matrix3d& cell_matrix) {
		std::vector<std::pair<Atom, double>> atoms_upper, atoms_lower;
		atoms_upper.reserve(4);
		atoms_lower.reserve(4);

		for (const Atom& corner : A_rel_to_B) {
			Atom corner_rel_to_COM { corner.m_atom_type, minimumImage(corner.m_position - COM_rel_to_B) };
			double angle { getAngle(cell_matrix*corner_rel_to_COM.m_position) };
			(corner_rel_to_COM.m_position[1] < 0 ? atoms_lower : atoms_upper).emplace_back(corner_rel_to_COM, angle);
		}

		std::pair<Atom, double> rightmost_corner_upper { atoms_upper.at(0) };
		for (const auto& corner : atoms_upper) {
			if (corner.first.m_position.x() > rightmost_corner_upper.first.m_position.x()) {
				rightmost_corner_upper = corner;
			}
		}
		auto result_upper = std::ranges::find_if(atoms_upper, [&](const std::pair<Atom, double>& pair) { 
			return pair.first.m_position.isApprox(rightmost_corner_upper.first.m_position);
		});
		std::ranges::swap(*result_upper, atoms_upper.front());

		std::pair<Atom, double> rightmost_corner_lower { atoms_lower.at(0) };
		for (const auto& corner : atoms_lower) {
			double dist { (corner.first.m_position - rightmost_corner_upper.first.m_position).squaredNorm() };
			if (dist < (rightmost_corner_lower.first.m_position - rightmost_corner_upper.first.m_position).squaredNorm()) {
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
		auto sort_by_angles = [](auto& arr) { 
			std::ranges::sort(arr.begin()+1, arr.end(), [](auto& pair1, auto& pair2) { 
				return pair1.second < pair2.second;
			});
		};

		sort_by_angles(atoms_upper);
		sort_by_angles(atoms_lower);

		return std::make_pair(atoms_upper, atoms_lower);
	}

public:
	friend void calculateLatticeConstant(std::vector<LocalUC>& local_UCs);

	double getLatticeConstant() const {
		return m_global_lattice_constant.value();
	}

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

	explicit LocalUC(const Atoms& A, const Atom& B, const Atoms& O, PhaseFactor phase_factor, DWType DW_type, const Eigen::Matrix3d& cell_matrix, double DW_center_x = 0.5, double tolerance = 1e-3) 
	: m_B_direct_pbc(B), m_metric(cell_matrix.transpose() * cell_matrix), m_phase_factor(phase_factor), m_type(DW_type)
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
				size_t i {};
				for (Atom& atom : temp){
					atom.m_position = minimumImage(atom.m_position - B.m_position);

				}
				return temp;
			}()
		};

		Position COM_rel_to_B = helper::getCOM(A_rel_to_B);
		m_COM_cart_nopbc = convertCoordinates(B.m_position + COM_rel_to_B, cell_matrix);

		setDomain(B.m_position + COM_rel_to_B, DW_center_x, tolerance);

		std::vector<std::pair<Atom, double>> atoms_upper, atoms_lower;
		atoms_upper.reserve(4);
		atoms_lower.reserve(4);


		if (DW_type == DWType::APB) {
			auto [upper, lower] = buildAPBCell(A_rel_to_B, B, COM_rel_to_B, cell_matrix);
			atoms_upper = upper;
			atoms_lower = lower;
		}
		else {
			auto [upper, lower] = buildTwinCell(A_rel_to_B, B, COM_rel_to_B, cell_matrix);
			atoms_upper = upper;
			atoms_lower = lower;
		}

		// fill A and O arrays
		auto fill_direct_A = [&](auto&& upper, auto&& lower){
			for (size_t i { 0 }; i < upper.size(); i++) {
				Atom first { std::move(lower[i].first)};
				Atom second { std::move(upper[i].first)};
				m_A_direct_pbc.emplace_back(std::move(first), std::move(second));
			}
		};

		m_A_direct_pbc.reserve(4);
		fill_direct_A(atoms_upper, atoms_lower);

		// sort O top, bottom, ....
		Atoms O_COM_centered { 
			[&](){
				Atoms temp { O };
				for (Atom& atom : temp){
					atom.m_position = minimumImage(atom.m_position - (B.m_position + COM_rel_to_B));
				}
				return temp; 
			}()
		};

		std::vector<std::pair<Atom, Atom>> O_B_centered_pairs { findOppositeO(O_COM_centered) };
		// wrap O back to direct with pbc
		wrapDirectCoordinates(O_B_centered_pairs, B.m_position + COM_rel_to_B);
		m_O_direct_pbc = O_B_centered_pairs;

		// get cartesian coordinates without pbc
		m_A_cart_nopbc.reserve(4);
		m_O_cart_nopbc.reserve(3);

		Atom ref_for_unwrap { B.m_atom_type, B.m_position + COM_rel_to_B };
		auto fill_cart = [&](const auto& direct, auto& cart) {
			for (const auto& pair : direct) {
				Atom first  { pair.first.m_atom_type,  get_cart_pos_nowrap(pair.first,  ref_for_unwrap, cell_matrix) };
				Atom second { pair.second.m_atom_type, get_cart_pos_nowrap(pair.second, ref_for_unwrap, cell_matrix) };
				cart.emplace_back(std::move(first), std::move(second));
			}
		};

		// save in cartesian without pbc
		fill_cart(m_A_direct_pbc, m_A_cart_nopbc);
		fill_cart(m_O_direct_pbc, m_O_cart_nopbc);
		m_B_cart_nopbc = Atom(m_B_direct_pbc.m_atom_type, get_cart_pos_nowrap(B, ref_for_unwrap, cell_matrix));

		// avg lattice constant
		const Atoms m_A_cart_nopbc_flattened { [&]() {
			Atoms temp; temp.reserve(8);
			for (auto& [upper, lower] : m_A_cart_nopbc) {
				temp.emplace_back(upper);
				temp.emplace_back(lower);
			}
			return temp;
		}() };

		std::vector<double> lattice_dists;
		lattice_dists.reserve(12);

		for (size_t i {}; i < m_A_cart_nopbc.size(); i++) {
			double dist;
			size_t j { (i+1)%m_A_cart_nopbc.size() };

			dist = (m_A_cart_nopbc.at(i).first.m_position - m_A_cart_nopbc.at(j).first.m_position).norm();
			lattice_dists.push_back(dist);
			dist = (m_A_cart_nopbc.at(i).second.m_position - m_A_cart_nopbc.at(j).second.m_position).norm();
			lattice_dists.push_back(dist);
			dist = (m_A_cart_nopbc.at(i).second.m_position - m_A_cart_nopbc.at(j).second.m_position).norm();
			lattice_dists.push_back(dist);
		}

		m_local_lattice_constant = std::accumulate(lattice_dists.begin(), lattice_dists.end(), 0.0) / lattice_dists.size();
	}

	void updateInitialOrientation(const std::tuple<Eigen::Quaterniond, short, Rotation>& orientation) {
		if (m_side == DWSide::left) {
			m_left_init_orientation = orientation;
		}
		else if (m_side == DWSide::right) {
			m_right_init_orientation = orientation;
		}
	}

	std::pair<std::tuple<Eigen::Quaterniond, short, Rotation>, std::tuple<Eigen::Quaterniond, short, Rotation>> getInitialOrientations() const {
		return std::make_pair(m_left_init_orientation.value(), m_right_init_orientation.value());
	}


	std::expected<void, std::string> setInitialOrientation(const std::tuple<Eigen::Quaterniond, short, Rotation>& orientation) { 
		// orientation short: perm num, Rotation: perm dir, Quaternion: rotation of permuted unit cell
		if (m_type == DWType::APB) {
			m_right_init_orientation = std::make_tuple(Eigen::Quaterniond(1, 0, 0, 0), 0, Rotation::None);
			m_left_init_orientation = m_right_init_orientation;
			return {};
		}

		if (m_side == DWSide::center) {
			return std::unexpected("Cell in DW center!"); 
		}

		if (m_left_init_orientation && m_right_init_orientation) {
			return std::unexpected("Initial Orientation already set!");
		}

		const Eigen::Quaterniond& quaternion { std::get<0>(orientation) };
		const short permutation_number { std::get<1>(orientation) };

		Eigen::Quaterniond rot_90_y { cos(M_PI/4), 0, sin(M_PI/4), 0 };
		const double angle { [&]() {
			Vector z_rotated { quaternion*Vector(0,0,1) };
			double angle { atan2(z_rotated.z(), z_rotated.x()) };
			return angle < 0 ? angle+2*M_PI : angle; // get the angle of rotated z axis around y axis
		}() };

		// local quadrant
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
				m_left_init_orientation = orientation;
				if (quadrant_2 || quadrant_4) {
					m_right_init_orientation = std::make_tuple(quaternion*rot_90_y.conjugate(), get_permutation_number(permutation_number-1), Rotation::left);
				}
				else {
					m_right_init_orientation = std::make_tuple(quaternion*rot_90_y, get_permutation_number(permutation_number+1), Rotation::left);
				}
			}
			else {
				m_right_init_orientation = orientation;
				if (quadrant_2 || quadrant_4) {
					m_left_init_orientation = std::make_tuple(quaternion*rot_90_y.conjugate(), get_permutation_number(permutation_number-1), Rotation::left);
				}
				else {
					m_left_init_orientation = std::make_tuple(quaternion*rot_90_y, get_permutation_number(permutation_number+1), Rotation::left);
				}
			}
		}
		else {
			if (m_side == DWSide::left) {
				m_left_init_orientation = orientation;
				if (quadrant_2 || quadrant_4) {
					m_right_init_orientation = std::make_tuple(quaternion*rot_90_y, get_permutation_number(permutation_number+1), Rotation::left);
				}
				else {
					m_right_init_orientation = std::make_tuple(quaternion*rot_90_y.conjugate(), get_permutation_number(permutation_number-1), Rotation::left);
				}
			}
			else {
				m_right_init_orientation = orientation;
				if (quadrant_2 || quadrant_4) {
					m_left_init_orientation = std::make_tuple(quaternion*rot_90_y, get_permutation_number(permutation_number+1), Rotation::left);
				}
				else {
					m_left_init_orientation = std::make_tuple(quaternion*rot_90_y.conjugate(), get_permutation_number(permutation_number-1), Rotation::left);
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
			Position pos { pair.first.m_position - m_B_cart_nopbc.m_position };

			if (pos.dot(axis) > 0) {
				Atom temp { pair.first };	
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
		centered_uc.m_COM_cart_nopbc.setZero();
		return centered_uc;
	}

	short getPhaseFactor() const {
		return m_phase_factor.value() == PhaseFactor::positive ? 1 : -1;
	}


	void calculateLocalOP() {
		// pass uncentered unit cell
		// centered at origin, ey into plane, sequence: Sr front lower left, Ti, O front, O bottom, O left
		const Eigen::Quaterniond& unit_quaternion { std::get<0>(getOrientation().value()) };
		const short permutation_number { std::get<1>(getOrientation().value()) };
		const UnitCell::Rotation permutation_direction { std::get<2>(getOrientation().value()) };

		Vector zeros { Vector::Zero() };
		Vector ex { 1, 0, 0 };
		Vector ey { 0, 1, 0 };
		Vector ez { 0, 0, 1 };

		Eigen::VectorXd phonon_pol_x(15); // { 0, 0, 0,   0, 0, 0,   0, 0, -1,   0, 1, 0,   0,  0, 0 }
		Eigen::VectorXd phonon_pol_y(15); // { 0, 0, 0,   0, 0, 0,   0, 0,  0,  -1, 0, 0,   0,  0, 1 }
		Eigen::VectorXd phonon_pol_z(15); // { 0, 0, 0,   0, 0, 0,   1, 0,  0,   0, 0, 0,   0, -1, 0 }

		phonon_pol_x << zeros, zeros, -ez, ey, zeros;
		phonon_pol_y << zeros, zeros, zeros, -ex, ez;
		phonon_pol_z << zeros, zeros, ex, zeros, -ey;

		phonon_pol_x *= std::sqrt(1.0/2.0); //.normalize();
		phonon_pol_y *= std::sqrt(1.0/2.0); //.normalize();
		phonon_pol_z *= std::sqrt(1.0/2.0); //.normalize();

		// rotate local centered UC into global frame
		UnitCell reference_UC { UnitCell(AtomType::Sr, AtomType::Ti, m_global_lattice_constant.value()) };
		reference_UC.RotateUC(Eigen::Quaterniond(1,0,0,0), permutation_number, permutation_direction);
		LocalUC local_UC_centered { getCenteredUC() };
		local_UC_centered.RotateUC(unit_quaternion.conjugate());

		double m_Sr { ATOM_MASSES_U.at(0) };
		double m_Ti { ATOM_MASSES_U.at(1) };
		double m_O { ATOM_MASSES_U.at(2) };
		double total_mass { m_Sr + m_Ti + 3*m_O };

		UnitCell::Displacements displacements { local_UC_centered - reference_UC };
		Vector Sr_displacement_mass_weighted { sqrt(m_Sr/total_mass)*displacements.m_A_displacements.at(0).first };
		Vector Ti_displacement_mass_weighted { sqrt(m_Ti/total_mass)*displacements.m_B_displacement };
		Vector O_front_displacement_mass_weighted { sqrt(m_O/total_mass)*displacements.m_O_displacements.at(0).first };
		Vector O_bottom_displacement_mass_weighted { sqrt(m_O/total_mass)*displacements.m_O_displacements.at(1).first };
		Vector O_left_displacement_mass_weighted { sqrt(m_O/total_mass)*displacements.m_O_displacements.at(2).first };

		Eigen::VectorXd mass_weighted_displacements(15);
		mass_weighted_displacements << Sr_displacement_mass_weighted,
			Ti_displacement_mass_weighted,
			O_front_displacement_mass_weighted,
			O_bottom_displacement_mass_weighted,
			O_left_displacement_mass_weighted;

		Vector phi {};
		phi.x() = mass_weighted_displacements.dot(phonon_pol_x);
		phi.y() = mass_weighted_displacements.dot(phonon_pol_y);
		phi.z() = mass_weighted_displacements.dot(phonon_pol_z);

		short phase_factor { getPhaseFactor() };

		m_local_OP_local_frame = phase_factor*phi;
		m_local_OP_global_frame = unit_quaternion*(phase_factor*phi);
	}

	void calculateLocalPolarization(/*const Eigen::Matrix3d& BEC_Sr, const Eigen::Matrix3d& BEC_Ti, const Eigen::Matrix3d& BEC_O*/) {
		const Eigen::Quaterniond& unit_quaternion { std::get<0>(getOrientation().value()) };
		const short permutation_number { std::get<1>(getOrientation().value()) };
		const UnitCell::Rotation permutation_direction { std::get<2>(getOrientation().value()) };

		UnitCell reference_rotated { AtomType::Sr, AtomType::Ti, m_global_lattice_constant.value()};
		reference_rotated.RotateUC(unit_quaternion, permutation_number, permutation_direction);
		LocalUC local_UC_centered { getCenteredUC() };

		double elementary_charge { 1.602176634e-19 };
		Displacements displacements { local_UC_centered - reference_rotated };

		auto rotate_displacements = [&unit_quaternion](std::vector<std::pair<Vector, Vector>>& displacements) {
			for (auto&& [first, second] : displacements) {
				first = unit_quaternion.conjugate()*first;
				second = unit_quaternion.conjugate()*second;
			}
		};

		rotate_displacements(displacements.m_A_displacements);
		displacements.m_B_displacement = unit_quaternion.conjugate()*displacements.m_B_displacement;
		rotate_displacements(displacements.m_O_displacements);

		double Z_Sr = 2.54, Z_Ti = 7.12, Z_Op = -5.66, Z_On = -2.0;

		Eigen::Matrix3d BEC_Sr;
		Eigen::Matrix3d BEC_Ti;
		Eigen::Matrix3d BEC_Ox;
		Eigen::Matrix3d BEC_Oy;
		Eigen::Matrix3d BEC_Oz;

		BEC_Sr << Z_Sr, 0, 0,
			0, Z_Sr, 0,
			0, 0, Z_Sr;

		BEC_Ti << Z_Ti, 0, 0,
			0, Z_Ti, 0,
			0, 0, Z_Ti;

		BEC_Ox << Z_Op, 0, 0,
			0, Z_On, 0,
			0, 0, Z_On;

		BEC_Oy << Z_On, 0, 0,
			0, Z_Op, 0,
			0, 0, Z_On;

		BEC_Oz << Z_On, 0, 0,
			0, Z_On, 0,
			0, 0, Z_Op;

		Vector polarization { Vector::Zero() };

		//std::println("localUC Oy coords {} {} {}, Oz coords {} {} {}, Ox coords {} {} {}, quaternion: w={} x={} y={} z={}", local_UC_centered.m_O_cart_nopbc.at(0).first.m_position.x(), local_UC_centered.m_O_cart_nopbc.at(0).first.m_position.y(), local_UC_centered.m_O_cart_nopbc.at(0).first.m_position.z(),
		//	   local_UC_centered.m_O_cart_nopbc.at(1).first.m_position.x(), local_UC_centered.m_O_cart_nopbc.at(1).first.m_position.y(), local_UC_centered.m_O_cart_nopbc.at(1).first.m_position.z(), local_UC_centered.m_O_cart_nopbc.at(2).first.m_position.x(), local_UC_centered.m_O_cart_nopbc.at(2).first.m_position.y(), local_UC_centered.m_O_cart_nopbc.at(2).first.m_position.z(),
		//	   unit_quaternion.x(), unit_quaternion.y(), unit_quaternion.z(), unit_quaternion.w());
		//std::println();

		polarization.array() += (BEC_Sr*displacements.m_A_displacements.at(0).first).array();
		polarization.array() += (BEC_Ti*displacements.m_B_displacement).array();
		polarization.array() += (BEC_Oy*displacements.m_O_displacements.at(0).first).array();
		polarization.array() += (BEC_Oz*displacements.m_O_displacements.at(1).first).array();
		polarization.array() += (BEC_Ox*displacements.m_O_displacements.at(2).first).array();

		polarization /= reference_rotated.m_cell_volume;

		m_local_polarization_local_frame = polarization;
		m_local_polarization_global_frame = unit_quaternion*polarization;
	}

};

inline void calculateLatticeConstant(std::vector<LocalUC>& local_UCs) {
	double sum = std::accumulate(local_UCs.begin(), local_UCs.end(), 0.0, [](double acc, const LocalUC& local_UC) {
		return acc + local_UC.m_local_lattice_constant;
	});
	
	local_UCs.at(0).m_global_lattice_constant = sum/local_UCs.size();
}


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
										const bool minimum_image = true,
										const std::optional<Eigen::Matrix3d> &cell_matrix = std::nullopt) {

	Vector dr { atom2.m_position - atom1.m_position };

	if (minimum_image) {
		dr.array() -= (dr.array() + 0.5 - 1e-12).floor();
	}

	return getSqDistance(dr, cell_matrix);
}

inline std::expected<NNIds, std::string> findNearestN(const Atom& reference_atom, 
													  const Atoms& atom_arr, 
													  const size_t n, 
													  const std::optional<size_t> exclude_idx,
													  const std::optional<Eigen::Matrix3d>& cell_matrix = std::nullopt, 
													  const bool wrap = true,
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

		if (wrap) {
			nearest_neighbors.emplace_back(idx++, getMinimumImageSqDistance(reference_atom, other_atom, true, cell_matrix));
		}
		else {
			nearest_neighbors.emplace_back(idx++, getMinimumImageSqDistance(reference_atom, other_atom, false, cell_matrix));
		}
	}

	auto minDistance = [](const auto &pair1, const auto &pair2) { 
		return pair1.second < pair2.second; 
	};


	if (wrap) {
		std::ranges::nth_element(nearest_neighbors, nearest_neighbors.begin() + n, minDistance);
		nearest_neighbors.resize(n);
	}
	else {
		std::ranges::partial_sort(nearest_neighbors, nearest_neighbors.begin()+n, minDistance);

		size_t id {};
		double cur_dist { nearest_neighbors.at(0).second };
		while (nearest_neighbors.at(id).second < cur_dist + 1.5) {
			id++;
		}
		nearest_neighbors.resize(id);
	}

	if (sort) {
		std::ranges::sort(nearest_neighbors.begin(), nearest_neighbors.begin() + n, [](const auto &p1, const auto &p2){
			return p1.second < p2.second;
		});
	}

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

inline std::vector<LocalUC::PhaseFactor> findPhaseFactor(const Atoms& B, const std::vector<NNIds> B_NNs) {
	// two coloring BFS algorithm to traverse the lattice as a spanning graph and setting the bloch wave phase factor accordingly
	std::vector<int> phase_factors(B.size(), 0);
	std::deque<size_t> queue;

	queue.push_back(0);
	phase_factors.at(0) = 1; // seed the initial phase factor
	while (!queue.empty()) {
		size_t current_parent_id { queue.front() };
		queue.pop_front();

		for (const auto& [current_child_id, dist] : B_NNs.at(current_parent_id)) {
			if (phase_factors.at(current_child_id) == 0) {
				queue.push_back(current_child_id);
				phase_factors.at(current_child_id) = -1*phase_factors.at(current_parent_id);
			}
				// exchange the continue with a new NN search without pbc first, store phase factor and b atom as pair, then full NN search and use that then to make the uc but with the phase factor in the pair 
				//throw std::runtime_error("Neighboring cells have the same phase_factor!");
			std::println("parent id {}, child id {}, parent sign {} child sign {}", current_parent_id, current_child_id, phase_factors.at(current_parent_id), phase_factors.at(current_child_id));
		}
	}

	std::vector<LocalUC::PhaseFactor> out; out.reserve(B.size());
	for (size_t i { }; i < B.size(); i++) {
		out.emplace_back(phase_factors.at(i) == 1 ? LocalUC::PhaseFactor::positive : LocalUC::PhaseFactor::negative);
	}

	return out;
}

inline Eigen::Quaterniond gradientDescent(const UnitCell& pristine_UC, const LocalUC &local_UC, 
										  const std::tuple<Eigen::Quaterniond, short, UnitCell::Rotation>& init_orientation, double step_size = 0.0005 ) {
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

	Eigen::Quaterniond initial_quaternion { std::get<0>(init_orientation).normalized() };
	Eigen::Quaterniond current_unit_quaternion { 1,0,0,0 };

	UnitCell pristine_UC_init { pristine_UC.getRotatedUC(initial_quaternion, std::get<1>(init_orientation), std::get<2>(init_orientation)) };
	LocalUC local_UC_centered { local_UC.getCenteredUC() };

	double cur_step_size { step_size };
	size_t cur_iter { };
	double cur_sq_dist { sq_dist(local_UC_centered - pristine_UC_init )};
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

	auto g = [&]() {
		Eigen::Quaterniond g;
		auto& g_coeff { g.coeffs() };

		for (size_t i { }; i < 4; i++) {
			const Eigen::Matrix3d grad_R_i { gradR(i) }; 
			double sum { };
			for (size_t j { }; j < 4; j++) {
				const std::pair<Atom, Atom>& pristine_atoms_A = pristine_UC_init.m_A_cart_nopbc[j];
				const std::pair<Atom, Atom>& local_atoms_A = local_UC_centered.m_A_cart_nopbc[j];

				sum += local_atoms_A.first.m_position.dot(grad_R_i*pristine_atoms_A.first.m_position);
				sum += local_atoms_A.second.m_position.dot(grad_R_i*pristine_atoms_A.second.m_position);
			}
			g_coeff[i] = -2*sum/(3.905*3.905);
		}

		return g;
	};

	auto lambda = [](const Eigen::Quaterniond& q, const Eigen::Quaterniond& grad_L, double cur_step_size) { 
		double a { (q.conjugate()*grad_L).w() };
		double b { grad_L.squaredNorm() };
		return (1 - cur_step_size*a - std::sqrt(std::pow(cur_step_size,2)*(std::pow(a,2) - b) + 1));
	};

	auto line_search = [&sq_dist, &lambda, &local_UC_centered, &pristine_UC_init]
		(double cur_sq_dist, const Eigen::Quaterniond& q, const Eigen::Quaterniond& grad_L, double step_size) {
			Eigen::Quaterniond best_q { q };
			double best_dist { cur_sq_dist };
			double cur_step_size { step_size*0.01 };
			for (size_t i { 0 }; i <= 100; i++) {
				double new_step_size = std::pow(1.1, i) * cur_step_size;
				double new_lambda { lambda(q, grad_L, new_step_size) };
				Eigen::Vector4d delta { new_step_size*grad_L.coeffs() + new_lambda*q.coeffs() };
				// forward search
				Eigen::Quaterniond new_q_f { q.coeffs() - delta };
				Eigen::Quaterniond new_q_b { q.coeffs() + delta };
				new_q_f.normalize();
				new_q_b.normalize();

				double new_sq_dist_f { sq_dist(local_UC_centered - pristine_UC_init.getRotatedUC(new_q_f)) };
				double new_sq_dist_b { sq_dist(local_UC_centered - pristine_UC_init.getRotatedUC(new_q_b)) };

				if (new_sq_dist_f < best_dist || new_sq_dist_b < best_dist) {
					if (new_sq_dist_f <= new_sq_dist_b) {
						best_q = new_q_f;
						best_dist = new_sq_dist_f;
					}
					else {
						best_q = new_q_b;
						best_dist = new_sq_dist_b;
					}
				}
			}
			return std::make_pair(best_q, best_dist);
		};

	size_t counter { };
	while(cur_iter++ < max_iter) {
		const Eigen::Quaterniond grad_L { g() };

		auto [new_unit_quaternion, new_sq_dist] = line_search(cur_sq_dist, current_unit_quaternion, grad_L, cur_step_size);

		if (new_sq_dist >= cur_sq_dist && counter++ < 5) {
			cur_step_size /= 5;
			continue;
		}
		else if (new_sq_dist >= cur_sq_dist && counter >= 5) {
			break;
		}

		current_unit_quaternion = new_unit_quaternion;
		cur_sq_dist = new_sq_dist;
	}

	std::cout << "Quaternion " << current_unit_quaternion << std::endl;
	std::println("{}", cur_sq_dist);
	return (current_unit_quaternion*initial_quaternion).normalized();
}

inline void findInitialOrientation(LocalUC& local_UC, double step_size) {
	const LocalUC pseudo_unit_cell_centered { local_UC.getCenteredUC() };

	// vector with the 4 possible corner permutations, each corresponding to a pi/2 rotation of the local z axis around the y+ axis
	const std::vector<UnitCell> pristine_UC_perm { [&local_UC]() {
		std::vector<UnitCell> temp; temp.reserve(4);
		UnitCell base { AtomType::Sr, AtomType::Ti, +1, { 1, 0, 0, 0 }, local_UC.getLatticeConstant() };
		for (size_t i { }; i < 4; i++) {
			UnitCell base_rot { base.getRotatedUC( Eigen::Quaterniond( 1, 0, 0, 0 ), static_cast<short>(i), UnitCell::Rotation::left) };
			temp.emplace_back(base_rot);
		}
		return temp;
	}() };

	std::tuple<double, Eigen::Quaterniond, size_t> best_uc { std::numeric_limits<double>::infinity(), Eigen::Quaterniond(1,0,0,0), 0 }; // dist front O and corresponding uc

	if (local_UC.m_type == DWType::APB) {
		std::tuple<Eigen::Quaterniond, short, UnitCell::Rotation> initial_orientation { { 1, 0, 0, 0 }, 0, UnitCell::Rotation::None };
		Eigen::Quaterniond unit_quaternion { gradientDescent(pristine_UC_perm.at(0), local_UC, initial_orientation, step_size) }; 

		std::get<0>(best_uc) = 0;
		std::get<1>(best_uc) = unit_quaternion;
		std::get<2>(best_uc) = 0;
	}
	else {
		for (std::size_t i { }; i<4; i++) {
			std::tuple<Eigen::Quaterniond, short, UnitCell::Rotation> initial_orientation { { 1, 0, 0, 0 }, 0, UnitCell::Rotation::None };
			// fit each permuted pristine uc, get q, measure O front and pick best, don't rotate when already permuted
			Eigen::Quaterniond unit_quaternion { gradientDescent(pristine_UC_perm.at(i), local_UC, initial_orientation, step_size) }; 

			const UnitCell pristine_rot { pristine_UC_perm.at(i).getRotatedUC(unit_quaternion) };
			const Position& pristine_UC_front_O { pristine_rot.m_O_cart_nopbc.at(0).first.m_position };
			const Position& pseudo_UC_front_O { pseudo_unit_cell_centered.m_O_cart_nopbc.at(0).first.m_position };
			double cur_front_O_sq_distance { (pristine_UC_front_O - pseudo_UC_front_O).squaredNorm() };

			// update best distance
			if (cur_front_O_sq_distance < std::get<0>(best_uc)) {
				std::get<0>(best_uc) = cur_front_O_sq_distance;
				std::get<1>(best_uc) = unit_quaternion;
				std::get<2>(best_uc) = i;
			}
		}
	}

	Eigen::Quaterniond unit_quaternion { std::get<1>(best_uc) };
	std::pair<short, UnitCell::Rotation> permutation = pristine_UC_perm.at(std::get<2>(best_uc)).getInitialPermutation();
	std::tuple<Eigen::Quaterniond, short, UnitCell::Rotation> final_orientation = std::make_tuple(unit_quaternion, permutation.first, permutation.second);

	std::expected<void, std::string> res = local_UC.setInitialOrientation(final_orientation);

	if (!res) {
		throw std::runtime_error("set initial Orientation failed");
	}

	local_UC.m_orientation = final_orientation;
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

// Add near the top of polcalc.hpp
struct POSCARData {
	Eigen::Matrix3d m_cell;              // columns are a, b, c
	Positions       m_positions_direct;  // fractional coordinates
	std::vector<std::string> m_symbols;
	std::vector<size_t>       m_counts;
};



// XDATCAR parser — drop into your project next to polcalc.hpp (same namespace)
// Produces one POSCAR-like frame per MD step in an XDATCAR file.
// Robust to VASP4/5 headers, with or without the symbols line, and supports
// both "Direct configuration = N" style and the single-frame style with a
// lone "Direct"/"Cartesian" line.


struct XDATFrame {
	long long step{};   // 1-based configuration index if available; otherwise sequential
	POSCARData data;    // cell, symbols, counts, positions (DIRECT)
};

struct XDATParseOptions {
	bool wrap_frac = false;   // wrap fractional coordinates to [0,1)
};

namespace xdatcar_detail {
inline std::vector<std::string> split_ws(const std::string &s) {
	std::vector<std::string> out; out.reserve(16);
	std::string cur; cur.reserve(s.size());
	for (char c: s) {
		if (std::isspace(static_cast<unsigned char>(c))) {
			if (!cur.empty()) { out.push_back(cur); cur.clear(); }
		} else cur.push_back(c);
	}
	if (!cur.empty()) out.push_back(cur);
	return out;
}
inline bool all_int(const std::vector<std::string>& v) {
	if (v.empty()) return false;
	for (auto &s: v) {
		char *e=nullptr; std::strtoll(s.c_str(), &e, 10); if (!(e && *e=='\0')) return false;
	}
	return true;
}
inline void wrap01(Eigen::Vector3d &f) {
	for (int i=0;i<3;++i) f[i] -= std::floor(f[i]);
}
inline std::string lower(std::string s){ for(char &c:s) c=(char)std::tolower((unsigned char)c); return s; }

// Try to parse lines like: "Direct configuration =   42" or "Cartesian configuration=7"
inline bool parse_config_header(const std::string& line, bool &is_direct, long long &step_out) {
	auto low = lower(line);
	if (low.find("configuration") == std::string::npos) return false;
	is_direct = low.find("direct") != std::string::npos;
	// find '=' and read an integer after it
	auto eq = low.find('=');
	if (eq != std::string::npos) {
		std::string tail = low.substr(eq+1);
		// strip spaces
		size_t i=0; while (i<tail.size() && std::isspace((unsigned char)tail[i])) ++i;
		long long v=0; bool any=false;
		for (; i<tail.size(); ++i) {
			if (std::isdigit((unsigned char)tail[i])) { any=true; v = v*10 + (tail[i]-'0'); }
			else break;
		}
		if (any) { step_out = v; return true; }
	}
	// If no explicit number, still treat as config header
	step_out = -1; return true;
}
}

inline std::expected<std::vector<XDATFrame>, std::string>
readXDATCAR(const std::string& filename, const XDATParseOptions& opt = {})
{
	using namespace xdatcar_detail;
	std::ifstream in(filename);
	if (!in) return std::unexpected("Failed loading file: " + filename);

	std::string line;
	// 1) Comment/title
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at comment");

	// 2) Scale
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at scale");
	double scale = 1.0; {
		auto toks = split_ws(line); if (toks.empty()) return std::unexpected("Bad scale line");
		scale = std::stod(toks[0]);
	}

	// 3) Lattice vectors (3 lines)
	auto read_vec = [&](Eigen::Vector3d& v)->bool{
		if (!std::getline(in, line)) return false; std::istringstream iss(line); return (bool)(iss>>v[0]>>v[1]>>v[2]); };
	Eigen::Vector3d a,b,c; if (!read_vec(a) || !read_vec(b) || !read_vec(c))
		return std::unexpected("Unexpected EOF at lattice vectors");
	Eigen::Matrix3d cell; cell.col(0) = scale*a; cell.col(1) = scale*b; cell.col(2) = scale*c;

	// 4) Symbols (optional VASP5) or counts (VASP4)
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF after lattice");
	auto toks = split_ws(line);

	std::vector<std::string> symbols;
	std::vector<size_t> counts;

	if (all_int(toks)) {
		// VASP4: no symbols line; fabricate using S1, S2, ...
		counts.reserve(toks.size());
		for (auto &t : toks) counts.push_back((size_t)std::stoull(t));
		symbols.reserve(counts.size());
		for (size_t i=0;i<counts.size();++i) symbols.push_back("S" + std::to_string(i+1));
	} else {
		symbols = toks; // VASP5 symbols line present
		if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at counts line");
		toks = split_ws(line);
		if (!all_int(toks)) return std::unexpected("Counts line is not all integers");
		counts.reserve(toks.size()); for (auto &t : toks) counts.push_back((size_t)std::stoull(t));
		if (symbols.size() != counts.size())
			return std::unexpected("Symbols and counts size mismatch");
	}

	// 5) Optional 'Selective dynamics' line — rarely appears in XDATCAR, skip if present
	std::streampos after_counts_pos = in.tellg();
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at coordinate header");
	std::string low = lower(line);
	if (low.starts_with("s")) { // Selective dynamics
		if (!std::getline(in, line)) return std::unexpected("Unexpected EOF after Selective dynamics");
		low = lower(line);
	}

	bool header_direct = false; // if a lone 'Direct' or 'Cartesian' line precedes the first frame
	long long step_from_header = -1;
	if (parse_config_header(line, header_direct, step_from_header)) {
		// e.g., "Direct configuration = 1"
	} else if (low.starts_with("d")) {
		header_direct = true;
	} else if (low.starts_with("c")) {
		header_direct = false;
	} else {
		// Not a coordinate header; rewind so the next read will pick this line again
		in.seekg(after_counts_pos);
	}

	const size_t n_atoms = [&]{ size_t n=0; for (auto x: counts) n+=x; return n; }();
	Eigen::Matrix3d invC = cell.inverse();

	std::vector<XDATFrame> frames;
	frames.reserve(64);

	auto read_one_frame = [&](bool as_direct, long long step_tag)->std::expected<Positions,std::string>{
		Positions pos; pos.reserve(n_atoms);
		for (size_t i=0;i<n_atoms; ++i) {
			if (!std::getline(in, line)) return std::unexpected("Unexpected EOF in coordinates at atom "+std::to_string(i));
			if (line.empty()) { --i; continue; }
			std::istringstream iss(line);
			double x,y,z; if (!(iss>>x>>y>>z)) return std::unexpected("Bad coordinate line at atom "+std::to_string(i));
			Eigen::Vector3d v(x,y,z);
			if (!as_direct) v = invC * v; // convert from Cartesian
			if (opt.wrap_frac) wrap01(v);
			pos.emplace_back(v);
		}
		return pos;
	};

	// Case A: file begins with a lone "Direct"/"Cartesian" line -> read first (and possibly only) frame
	if (low.starts_with("d") || low.starts_with("c") || step_from_header>=0) {
		bool as_direct = (step_from_header>=0) ? header_direct : header_direct; // same flag
		long long step = (step_from_header>=0) ? step_from_header : 1LL;
		auto pos = read_one_frame(as_direct, step);
		if (!pos) return std::unexpected(pos.error());
		POSCARData pd; pd.m_cell = cell; pd.m_symbols = symbols; pd.m_counts = counts; pd.m_positions_direct = std::move(*pos);
		frames.push_back(XDATFrame{step, std::move(pd)});
	}

	// Case B: additional frames marked by configuration lines
	while (true) {
		std::streampos here = in.tellg();
		if (!std::getline(in, line)) break; // EOF
		if (line.empty()) continue;
		bool as_direct=false; long long step=-1;
		if (!parse_config_header(line, as_direct, step)) {
			// If we read something else, rewind and stop (XDATCAR usually has only config lines after first frame)
			in.seekg(here); break;
		}
		auto pos = read_one_frame(as_direct, step);
		if (!pos) return std::unexpected(pos.error());
		POSCARData pd; pd.m_cell = cell; pd.m_symbols = symbols; pd.m_counts = counts; pd.m_positions_direct = std::move(*pos);
		frames.push_back(XDATFrame{ step>=0 ? step : (long long)frames.size()+1, std::move(pd)});
	}

	// If we never hit any header and didn't load a frame yet, try to read a single frame directly (rare)
	if (frames.empty()) {
		auto pos = read_one_frame(true, 1);
		if (!pos) return std::unexpected(pos.error());
		POSCARData pd; pd.m_cell = cell; pd.m_symbols = symbols; pd.m_counts = counts; pd.m_positions_direct = std::move(*pos);
		frames.push_back(XDATFrame{1, std::move(pd)});
	}

	return frames;
}

inline std::expected<std::vector<POSCARData>, std::string>
readXDATCARAsPOSCARFrames(const std::string& filename, const XDATParseOptions& opt = {})
{
	auto fr = readXDATCAR(filename, opt);
	if (!fr) return std::unexpected(fr.error());
	std::vector<POSCARData> out; out.reserve(fr->size());
	for (auto &f : *fr) out.push_back(std::move(f.data));
	return out;
}

// Robust VASP4/5 parser (handles optional symbols line and Selective dynamics)
inline std::expected<POSCARData, std::string>
readPOSCAR(const std::string& filename)
{
	std::ifstream in(filename);
	if (!in) return std::unexpected("Failed loading file: " + filename);

	auto split = [](const std::string& s){
		std::istringstream iss(s);
		std::vector<std::string> t; for (std::string w; iss>>w;) t.push_back(w);
		return t;
	};
	auto all_int = [](const std::vector<std::string>& v){
		if (v.empty()) return false;
		return std::ranges::all_of(v, [](const std::string& x){
			char* e=nullptr; std::strtoll(x.c_str(), &e, 10); return e && *e=='\0';
		});
	};

	std::string line;
	// 1) comment
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at comment");
	// 2) scale
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at scale");
	double scale = std::stod(split(line).at(0));

	auto read_vec = [&](Eigen::Vector3d& v)->bool{
		if (!std::getline(in, line)) return false;
		std::istringstream iss(line);
		return static_cast<bool>(iss >> v[0] >> v[1] >> v[2]);
	};

	Eigen::Vector3d a,b,c;
	if (!read_vec(a) || !read_vec(b) || !read_vec(c))
		return std::unexpected("Unexpected EOF at lattice vectors");

	// Build cell with lattice vectors as COLUMNS
	Eigen::Matrix3d cell;
	cell.col(0) = scale * a;
	cell.col(1) = scale * b;
	cell.col(2) = scale * c;

	// 6) symbols or counts
	if (!std::getline(in, line)) return std::unexpected("Unexpected EOF at symbols/counts");
	auto toks = split(line);
	std::vector<std::string> symbols;
	std::vector<size_t> counts;

	if (all_int(toks)) {
		// VASP4: counts directly
		for (auto& t : toks) counts.push_back(static_cast<size_t>(std::stoll(t)));
	} else {
		symbols = toks;
		if (!std::getline(in, line)) return std::unexpected("Missing counts line");
		auto cts = split(line);
		if (!all_int(cts)) return std::unexpected("Counts line is not integers");
		for (auto& t : cts) counts.push_back(static_cast<size_t>(std::stoll(t)));
	}

	// Optional "Selective dynamics"
	std::streampos before_coord_type = in.tellg();
	if (!std::getline(in, line)) return std::unexpected("Missing coordinate type");
	{
		auto low = line; std::ranges::transform(low, low.begin(), ::tolower);
		if (!(low.starts_with("d") || low.starts_with("c"))) {
			// assume this was "Selective dynamics", read the real coord-type next
			if (!std::getline(in, line)) return std::unexpected("Missing coordinate type after Selective dynamics");
		}
	}
	std::string coordtype = line;
	std::string low = coordtype; std::ranges::transform(low, low.begin(), ::tolower);
	bool direct = low.starts_with("d"); // "Direct" or "Fractional"

	size_t n_atoms = 0; for (auto n : counts) n_atoms += n;

	Positions pos; pos.reserve(n_atoms);
	for (size_t i = 0; i < n_atoms; ++i) {
		if (!std::getline(in, line)) return std::unexpected("Unexpected EOF in coordinates");
		std::istringstream iss(line);
		double x,y,z; 
		if (!(iss >> x >> y >> z))
			return std::unexpected("Bad coordinate line at atom " + std::to_string(i));
		pos.emplace_back(x,y,z); // read as given
	}

	// Convert to DIRECT if needed
	if (!direct) {
		// r_dir = C^{-1} * r_cart
		Eigen::Matrix3d invC = cell.inverse();
		for (auto& p : pos) p = invC * p;
	}

	POSCARData out;
	out.m_cell = cell;
	out.m_positions_direct = std::move(pos);
	out.m_symbols = std::move(symbols);
	out.m_counts = std::move(counts);
	return out;
}

inline std::expected<AtomPositions, std::string> sortPositionsByType(const Positions& positions, 
																	 const size_t N_Sr, 
																	 const size_t N_Ti, 
																	 const size_t N_O) {
	AtomPositions atom_positions(N_Sr, N_Ti, N_O);
	std::ranges::copy(positions 
				   | std::views::transform([](const Position& positions){ return Atom(AtomType::Sr, positions); }) 
				   | std::views::take(N_Sr), std::back_inserter(atom_positions.m_A));
	std::ranges::copy(positions 
				   | std::views::transform([](const Position& positions){ return Atom(AtomType::Ti, positions); }) 
				   | std::views::drop(N_Sr) | std::views::take(N_Ti), std::back_inserter(atom_positions.m_B));
	std::ranges::copy(positions 
				   | std::views::transform([](const Position& positions){ return Atom(AtomType::O, positions); }) 
				   | std::views::drop(N_Sr+N_Ti) 
				   | std::views::take(N_O), std::back_inserter(atom_positions.m_O));

	if (atom_positions.m_A.size() != N_Sr) {
		return std::unexpected("m_Sr positions and N_Sr differ");
	};
	if (atom_positions.m_B.size() != N_Ti) {
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
																		  const bool wrap) {

	std::vector<NNIds> nearest_neighbors;
	nearest_neighbors.reserve(ref_atoms.size());

	size_t atom_id { 0 };
	for (const Atom& ref_atom : ref_atoms) {

		std::optional<size_t> exclude_idx { std::nullopt };
		if (&atoms == &ref_atoms) {
			exclude_idx = atom_id;
		}

		auto res = helper::findNearestN(ref_atom, atoms, n, exclude_idx, cell_matrix, wrap, false) 
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
												   const std::vector<helper::LocalUC::PhaseFactor> phase_factors, DWType DW_type, 
												   Eigen::Matrix3d& cell_matrix) {

	std::vector<helper::LocalUC> local_UCs;
	local_UCs.reserve(B.size());

	const auto B_ids = std::ranges::views::iota(size_t {0}, B.size()) ;
	for (const auto& [A_NNIds, B_id, O_NNIds, phase_factor] : std::ranges::views::zip(A_NNIds_all, B_ids, O_NNIds_all, phase_factors)) {

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

		local_UCs.emplace_back(std::move(A_local), std::move(B_local), std::move(O_local), phase_factor, DW_type, cell_matrix);
	}

	calculateLatticeConstant(local_UCs);

	return local_UCs;
}


inline std::pair<ObservableData, ObservableData> calculateObservable(const std::vector<helper::LocalUC>& local_UCs, double threshold = 1 /*in angstroem*/) { //different behavior for either P or OP
	std::vector<helper::LocalUC> local_UCs_cp { local_UCs };
	std::ranges::sort(local_UCs_cp, [](const auto& lhs, const auto& rhs) {
		return lhs.m_B_cart_nopbc.m_position.x() < rhs.m_B_cart_nopbc.m_position.x();
	});

	std::vector<bool> used(local_UCs.size(), false);
	std::vector<std::pair<double, Vectors>> bins_OP; // storing x value (bins) and OP of the Local UC at that position
	std::vector<std::pair<double, Vectors>> bins_polarization; // storing x value (bins) and Polarization of the Local UC at that position

	// sorting bins
	while (true) {
		auto it_current = std::ranges::find(used, false);
		if (it_current == used.end()) {
			break;
		}

		size_t current_id { static_cast<size_t>(std::distance(used.begin(), it_current)) };

		std::pair<double, Vectors> current_bin_OP;
		std::pair<double, Vectors> current_bin_polarization;
		double current_pos { local_UCs_cp.at(current_id).m_COM_cart_nopbc.x()};
		current_bin_OP.second.emplace_back(local_UCs_cp.at(current_id).m_local_OP_global_frame.value());
		current_bin_polarization.second.emplace_back(local_UCs_cp.at(current_id).m_local_polarization_global_frame.value()); 
		*it_current = true;

		double current_bin_center { current_pos };
		for (size_t i { current_id }; i < used.size(); i++) {
			if (used.at(i)) {
				continue;
			}

			double new_pos { local_UCs_cp.at(i).m_B_cart_nopbc.m_position.x() };
			double diff_x { std::abs(current_pos - new_pos) };

			if (diff_x > threshold) {
				break; // because nothing after would fall into the bin anyway
			}

			current_bin_OP.second.emplace_back(local_UCs_cp.at(i).m_local_OP_global_frame.value());
			current_bin_polarization.second.emplace_back(local_UCs_cp.at(i).m_local_polarization_global_frame.value());
			used.at(i) = true;
			current_bin_center += new_pos;
		}

		current_bin_OP.first = current_bin_center / current_bin_OP.second.size();
		current_bin_polarization.first = current_bin_center / current_bin_polarization.second.size();
		bins_OP.emplace_back(current_bin_OP);
		bins_polarization.emplace_back(current_bin_polarization);
	}

	ObservableData observable_OP(bins_OP.size());
	ObservableData observable_polarization(bins_OP.size());

	for (auto&& [pair_OP, pair_polarization]: std::ranges::views::zip(bins_OP, bins_polarization)) {
		observable_OP.m_bin_center.emplace_back(pair_OP.first);
		observable_polarization.m_bin_center.emplace_back(pair_polarization.first);
	}

	// calculate average and variance
	auto avg = [&](const auto& bins, ObservableData& observable) {
		for (const auto& [bin_center, obs] : bins) {
			Vector bin_avg { Vector::Zero() };
			for (const auto& vec : obs) {
				bin_avg += vec;	
			}
			bin_avg /= obs.size();
			observable.m_observable_average.emplace_back(std::move(bin_avg));
		}
	};

	auto var = [&](const auto& bins, ObservableData& observable) {
		for (const auto& [bin_data, avg] : std::ranges::views::zip(bins, observable.m_observable_average)) {
			Vector bin_var { Vector::Zero() };
			for (const auto& obs : bin_data.second) {
				Vector diff { obs - avg };
				bin_var += (diff.array() * diff.array()).matrix();
			}
			bin_var /= bin_data.second.size() - 1;
			observable.m_observable_variance.emplace_back(std::move(bin_var));
		}
	};

	avg(bins_OP, observable_OP);
	avg(bins_polarization, observable_polarization);
	var(bins_OP, observable_OP);
	var(bins_polarization, observable_polarization);

	return std::make_pair(observable_OP, observable_polarization);

};

inline void calculateLocalObservables(std::vector<helper::LocalUC>& local_UCs, double step_size, bool OP = true, bool polarization = true) {
	// write a custom find/set initial orientation function for APBs 
	std::vector<size_t> DW_centers_init_ids; // containing all center DWs picked befor local z axis could be determined
	DW_centers_init_ids.reserve(20);

	helper::UnitCell pristine_UC_sp { AtomType::Sr, AtomType::Ti, 1, { 1, 0, 0, 0 }, local_UCs.at(0).getLatticeConstant() }; // sigma +1 UC
	helper::UnitCell pristine_UC_sn { AtomType::Sr, AtomType::Ti, -1, { 1, 0, 0, 0 }, local_UCs.at(0).getLatticeConstant() }; // sigma -1 UC

	auto getUnitCellData = [&](helper::LocalUC& local_UC) {
		helper::LocalUC local_UC_centered { local_UC.getCenteredUC() };
		auto local_UC_initial = local_UC.getInitialOrientation().value();
		Eigen::Quaterniond best_quaternion { helper::gradientDescent(pristine_UC_sp, local_UC, local_UC_initial, step_size) };
		std::tuple<Eigen::Quaterniond, short, helper::UnitCell::Rotation> orientation = std::make_tuple(best_quaternion, std::get<1>(local_UC_initial), std::get<2>(local_UC_initial));
		local_UC.m_orientation = orientation;
		local_UC.updateInitialOrientation(orientation); // use orientation of last cell on same side as initial

		helper::UnitCell pristine_UC_sp_rot { pristine_UC_sp.getRotatedUC(best_quaternion) };
		helper::UnitCell pristine_UC_sn_rot { pristine_UC_sn.getRotatedUC(best_quaternion) };
		Position front_O_sp { pristine_UC_sp_rot.m_O_cart_nopbc.at(0).first.m_position };
		Position front_O_sn { pristine_UC_sn_rot.m_O_cart_nopbc.at(0).first.m_position };
		Position local_UC_front_O { local_UC_centered.m_O_cart_nopbc.at(0).first.m_position };

		double displacement_sp { (front_O_sp - local_UC_front_O).squaredNorm() };
		double displacement_sn { (front_O_sn - local_UC_front_O).squaredNorm() };

		displacement_sp <= displacement_sn ? local_UC.m_O_rot_sign = 1 : local_UC.m_O_rot_sign = -1;

		local_UC.sortOs(best_quaternion);

		// calculate OP and Polarization
		if (OP) {
			local_UC.calculateLocalOP();
		}

		if (polarization) {
			local_UC.calculateLocalPolarization();
		}
	};

	bool is_first { true };
	for (size_t i { }; i < local_UCs.size(); i++) {
		if (is_first && local_UCs.at(i).m_side == helper::LocalUC::DWSide::center) { // avoid artifacts due to different O placement in DW center cells
			DW_centers_init_ids.push_back(i);
			continue;
		}

		if (is_first) {
			helper::findInitialOrientation(local_UCs.at(i), step_size);
			getUnitCellData(local_UCs.at(i));
			is_first = false;
		}
		else {
			getUnitCellData(local_UCs.at(i));
		};
	}

	for (size_t i : DW_centers_init_ids) {
		getUnitCellData(local_UCs.at(i));
	}
}

// add append option
inline void write(std::string filename, const ObservableData& data) {
	std::ofstream out { filename };
	if (!out) {
		throw std::runtime_error("failed to open file");
	}

	out.setf(std::ios::scientific);
	out << std::setprecision(8);

	for (const auto& [center, avg, var] : std::ranges::views::zip(data.m_bin_center, data.m_observable_average, data.m_observable_variance)) {
		out << center << ' ' << avg.x() << ' ' << avg.y() << ' ' << avg.z()
			<< ' ' << var.x() << ' ' << var.y() << ' ' << var.z() << '\n';
	}
}
}
