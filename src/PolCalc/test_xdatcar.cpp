#include "polcalc.hpp"

using namespace PolCalc;

int main() {
    auto xdatcar = readXDATCAR("./test/XDATCAR").value().at(0); // or "POSCAR"
	auto poscar = xdatcar.data;
    Positions positions_raw = poscar.m_positions_direct;

    // If you want to derive N_Sr/N_Ti/N_O from POSCAR instead of hardcoding:
    size_t N_Sr = poscar.m_counts.at(0), N_Ti = poscar.m_counts.at(1), N_O = poscar.m_counts.at(2);
    auto atoms = sortPositionsByType(positions_raw, N_Sr, N_Ti, N_O).value();

    Eigen::Matrix3d cell_matrix = poscar.m_cell; // <- critical

    auto A_NNs = getNearestNeighbors(atoms.m_A, atoms.m_B, 8, cell_matrix).value();
    auto O_NNs = getNearestNeighbors(atoms.m_O, atoms.m_B, 6, cell_matrix).value();
    auto B_NNs = getNearestNeighbors(atoms.m_B, atoms.m_B, 6, cell_matrix).value();
    auto B_NNs_no_wrap = getNearestNeighbors(atoms.m_B, atoms.m_B, 6, cell_matrix, false).value(); // for BFS phase factor

	auto phase_factors = helper::findPhaseFactor(atoms.m_B, B_NNs_no_wrap);

	auto local_UCs = createLocalUCs(atoms.m_A, atoms.m_B, atoms.m_O, A_NNs, O_NNs, phase_factors, PolCalc::DWType::APB, cell_matrix);

	calculateLocalObservables(local_UCs, 0.005);
	auto obs = calculateObservable(local_UCs, 0.25);

	write("example/OP/op.out", obs.first);
	write("example/POL/pol.out", obs.second);
}
