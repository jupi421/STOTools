// test_nn.cpp
#include <polcalc.hpp>
#include <Eigen/Core>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace PolCalc;

static constexpr const char* name = "./POSCAR_sto_bulk"; // adjust path as needed

static Atoms sliceToAtoms(const Positions& P, size_t start, size_t count, AtomType t) {
    Atoms out; out.reserve(count);
    for (size_t i = 0; i < count; ++i) out.emplace_back(t, P.at(start + i));
    return out;
}

void testFileReader() {
    Positions pos = loadPosFromFile(name, /*head=*/8);
    assert(pos.size() == 100 && "Expected 100 fractional positions (20 Sr, 20 Ti, 60 O)");
}

void testNearestNeighbors_SrAroundTi_frac() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 20, N_Ti = 20, N_O = 60;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Sr = sliceToAtoms(pos, 0,               N_Sr, AtomType::Sr);
    Atoms Ti = sliceToAtoms(pos, N_Sr,            N_Ti, AtomType::Ti);

    const size_t n = 8; // 8 A-site corners around each Ti in your layout
    auto nnExp = getNearestNeighbors(Sr, Ti, n, std::nullopt, /*sort=*/true);
    assert(nnExp.has_value());
    const auto& nn = *nnExp;
    assert(nn.size() == N_Ti);

    // For your grid: Δx=0.1, Δy=0.25, Δz=0.25 → d^2 = 0.01 + 0.0625 + 0.0625 = 0.135
    const double expected_d2 = 0.01 + 0.0625 + 0.0625; // 0.135
    for (const auto& lst : nn) {
        assert(lst.size() == n);
        for (const auto& [idx, d2] : lst) {
            (void)idx;
            assert(std::abs(d2 - expected_d2) < 1e-12 && "Sr corners should be equidistant from Ti");
        }
    }
}

void testNearestNeighbors_OAroundTi_frac() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 20, N_Ti = 20, N_O = 60;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Ti = sliceToAtoms(pos, N_Sr,            N_Ti, AtomType::Ti);
    Atoms O  = sliceToAtoms(pos, N_Sr + N_Ti,     N_O,  AtomType::O);

    const size_t n = 6; // octahedral O neighbors
    auto nnExp = getNearestNeighbors(O, Ti, n, std::nullopt, /*sort=*/true);
    assert(nnExp.has_value());
    const auto& nn = *nnExp;
    assert(nn.size() == N_Ti);

    for (const auto& lst : nn) {
        assert(lst.size() == n);
        size_t c001 = 0, c00625 = 0;
        for (const auto& [idx, d2] : lst) {
            (void)idx;
            if (std::abs(d2 - 0.01)   < 1e-12) ++c001;   // Δx = 0.1
            else if (std::abs(d2 - 0.0625) < 1e-12) ++c00625; // Δy or Δz = 0.25
            else assert(false && "Unexpected Ti–O distance (fractional metric)");
        }
        assert(c001 == 2 && c00625 == 4);
    }
}

void testSelfPool_TiExcludesSelf_frac() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 20, N_Ti = 20, N_O = 60;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Ti = sliceToAtoms(pos, N_Sr, N_Ti, AtomType::Ti);

    // self-pool: same Atoms instance for atoms and ref_atoms -> must exclude self
    const size_t n = 1;
    auto nnExp = getNearestNeighbors(Ti, Ti, n, std::nullopt, /*sort=*/true);
    assert(nnExp.has_value());
    const auto& nn = *nnExp;
    assert(nn.size() == N_Ti);

    for (size_t i = 0; i < N_Ti; ++i) {
        const auto& lst = nn[i];
        assert(lst.size() == n);
        assert(lst.front().first != i && "Self index must be excluded");
        // nearest Ti along x in your grid: Δ=0.2 → d^2=0.04
        assert(std::abs(lst.front().second - 0.04) < 1e-12);
    }
}

void testBadInputs_raiseExpectedErrors() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 20, N_Ti = 20, N_O = 60;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Sr = sliceToAtoms(pos, 0,    N_Sr, AtomType::Sr);
    Atoms Ti = sliceToAtoms(pos, N_Sr, N_Ti, AtomType::Ti);

    // n == 0
    {
        auto nnExp = getNearestNeighbors(Sr, Ti, 0, std::nullopt, true);
        assert(!nnExp.has_value());
    }
    // n > pool size
    {
        auto nnExp = getNearestNeighbors(Sr, Ti, N_Sr + 1, std::nullopt, true);
        assert(!nnExp.has_value());
    }
    // self-pool capacity: n > size-1
    {
        auto nnExp = getNearestNeighbors(Ti, Ti, N_Ti, std::nullopt, true);
        assert(!nnExp.has_value());
    }
}

int main() {
    testFileReader();
    testNearestNeighbors_SrAroundTi_frac();
    testNearestNeighbors_OAroundTi_frac();
    testSelfPool_TiExcludesSelf_frac();
    testBadInputs_raiseExpectedErrors();
    std::cout << "All NN tests (fractional) passed ✅\n";
    return 0;
}
