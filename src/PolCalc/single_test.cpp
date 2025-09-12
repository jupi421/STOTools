#include "polcalc.hpp"

using namespace PolCalc;

int main() {
	auto poscar = readPOSCAR("./test/POSCAR2").value(); // or "POSCAR"
	Positions positions_raw = poscar.m_positions_direct;

	// If you want to derive N_Sr/N_Ti/N_O from POSCAR instead of hardcoding:
	// size_t N_Sr = poscar.m_counts.at(0), N_Ti = poscar.m_counts.at(1), N_O = poscar.m_counts.at(2);
    auto atoms = sortPositionsByType(positions_raw, 800, 800, 2400).value();

    Eigen::Matrix3d cell_matrix = poscar.m_cell; // <- critical

    auto A_NNs = getNearestNeighbors(atoms.m_A, atoms.m_B, 8, cell_matrix).value();
    auto O_NNs = getNearestNeighbors(atoms.m_O, atoms.m_B, 6, cell_matrix).value();
    auto B_NNs = getNearestNeighbors(atoms.m_B, atoms.m_B, 6, cell_matrix).value();

    //for (size_t i = 0; i < B_NNs.size(); ++i) {
    //    std::println("reference index: {}", i);
    //    for (size_t j = 0; j < B_NNs.at(i).size(); ++j) {
    //        const auto& [nn_idx, d2] = B_NNs.at(i).at(j);
    //        std::println("NN index: {}, distance to ref {}", nn_idx, std::sqrt(d2));
    //    }
    //    std::println("");
    //}

	auto phase_factors = helper::findPhaseFactor(atoms.m_B, B_NNs);

	auto local_UCs = createLocalUCs(atoms.m_A, atoms.m_B, atoms.m_O, A_NNs, O_NNs, phase_factors, PolCalc::DWType::HT, cell_matrix);
	helper::UnitCell pristine_UC { AtomType::Sr, AtomType::O, +1, { 1, 0, 0, 0 } };
	helper::LocalUC local_UC { local_UCs.at(0).getCenteredUC() };

	auto sq_dist = [](helper::UnitCell::Displacements displacements) {
		double sq_dist { };
		for (const auto& pair : displacements.m_A_displacements) {
			sq_dist += pair.first.squaredNorm();
			sq_dist += pair.second.squaredNorm();
		}
		return sq_dist;
	};

	// quaternion 1deg around y
	double angle { M_PI/2 / 180 };
	Eigen::Quaterniond q { cos(angle/2), 0, sin(angle/2), 0 };

	std::ofstream out { "loss.dat" };
	if (!out) {
		throw std::runtime_error("failed to open file");
	}

	out.setf(std::ios::scientific);
	out << std::setprecision(8);

	
	double dist { sq_dist(local_UC - pristine_UC) };
	out << dist;
	for (size_t i { }; i < 720; i++) {
		pristine_UC.RotateUC(q);
		dist = sq_dist(local_UC - pristine_UC);
		out << dist << '\n';
	}
}
