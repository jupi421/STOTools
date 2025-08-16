// test_twinwall.cpp
#include <polcalc.hpp>
#include <Eigen/Core>
#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace PolCalc;

// Set this to your twin-wall POSCAR filename
static constexpr const char* name = "./POSCAR_twinwall";

// ---- helpers ----
static inline PolCalc::Atoms sliceToAtoms(const PolCalc::Positions& P, size_t start, size_t count, AtomType t) {
    Atoms out; out.reserve(count);
    for (size_t i = 0; i < count; ++i) out.emplace_back(t, P.at(start + i));
    return out;
}
static inline bool is_sorted_nondec(const std::vector<std::pair<size_t,double>>& v) {
    for (size_t i = 1; i < v.size(); ++i) if (v[i-1].second > v[i].second) return false;
    return true;
}
static inline void dump_first(const char* label,
                              const std::vector<std::vector<std::pair<size_t,double>>>& nn,
                              size_t centers_to_show = 2)
{
    std::cout << "\n" << label << " (first " << centers_to_show << " centers):\n";
    for (size_t i = 0; i < std::min(centers_to_show, nn.size()); ++i) {
        std::cout << "Center " << i << ":\n";
        for (size_t j = 0; j < nn[i].size(); ++j) {
            std::cout << "  #" << j << " idx=" << nn[i][j].first
                      << " d2_frac=" << std::setprecision(12) << nn[i][j].second << "\n";
        }
    }
}

// ---- tests ----
void testFileReader_twinwall() {
    // POSCAR header is 8 lines; coordinates after that.
    Positions pos = loadPosFromFile(name, /*head=*/8);
    assert(pos.size() == 480 && "Expected 480 positions (96 Sr, 96 Ti, 288 O)");
}

void testNN_SrAroundTi_twinwall() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 96, N_Ti = 96, N_O = 288;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Sr = sliceToAtoms(pos, 0,             N_Sr, AtomType::Sr);
    Atoms Ti = sliceToAtoms(pos, N_Sr,          N_Ti, AtomType::Ti);

    const size_t n = 8;
    auto nnExp = getNearestNeighbors(Sr, Ti, n, std::nullopt, /*sort=*/true);
    assert(nnExp.has_value());
    const auto& nn = *nnExp;
    assert(nn.size() == N_Ti);
    for (const auto& lst : nn) {
        assert(lst.size() == n);
        assert(is_sorted_nondec(lst));
        // fractional sanity: nearest Sr corners should be well below 0.25 in d^2
        assert(lst.back().second < 0.25);
    }
    dump_first("Sr (pool) around Ti (centers), n=8", nn);
}

void testNN_OAroundTi_twinwall() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 96, N_Ti = 96, N_O = 288;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Ti = sliceToAtoms(pos, N_Sr,          N_Ti, AtomType::Ti);
    Atoms O  = sliceToAtoms(pos, N_Sr + N_Ti,   N_O,  AtomType::O);

    const size_t n = 6;
    auto nnExp = getNearestNeighbors(O, Ti, n, std::nullopt, /*sort=*/true);
    assert(nnExp.has_value());
    const auto& nn = *nnExp;
    assert(nn.size() == N_Ti);
    for (const auto& lst : nn) {
        assert(lst.size() == n);
        assert(is_sorted_nondec(lst));
        // O octahedron in frac metric: should be small; keep a loose sanity bound
        assert(lst.back().second < 0.20);
    }
    dump_first("O (pool) around Ti (centers), n=6", nn);
}

void testSelfPool_Ti_excludes_self_twinwall() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 96, N_Ti = 96, N_O = 288;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Ti = sliceToAtoms(pos, N_Sr, N_Ti, AtomType::Ti);

    const size_t n = 1;
    auto nnExp = getNearestNeighbors(Ti, Ti, n, std::nullopt, /*sort=*/true);
    assert(nnExp.has_value());
    const auto& nn = *nnExp;
    assert(nn.size() == N_Ti);
    for (size_t i = 0; i < N_Ti; ++i) {
        assert(nn[i].size() == n);
        assert(nn[i].front().first != i && "Self index must be excluded");
        assert(nn[i].front().second >= 0.0);
    }
}

void testBadInputs_twinwall() {
    Positions pos = loadPosFromFile(name, 8);
    const size_t N_Sr = 96, N_Ti = 96, N_O = 288;
    assert(pos.size() == N_Sr + N_Ti + N_O);

    Atoms Sr = sliceToAtoms(pos, 0,    N_Sr, AtomType::Sr);
    Atoms Ti = sliceToAtoms(pos, N_Sr, N_Ti, AtomType::Ti);

    // n == 0
    {
        auto r = getNearestNeighbors(Sr, Ti, 0, std::nullopt, true);
        assert(!r.has_value());
    }
    // n > pool size
    {
        auto r = getNearestNeighbors(Sr, Ti, N_Sr + 1, std::nullopt, true);
        assert(!r.has_value());
    }
    // self-pool: n > size-1
    {
        auto r = getNearestNeighbors(Ti, Ti, N_Ti, std::nullopt, true);
        assert(!r.has_value());
    }
}

int main() {
    testFileReader_twinwall();
    testNN_SrAroundTi_twinwall();
    testNN_OAroundTi_twinwall();
    testSelfPool_Ti_excludes_self_twinwall();
    testBadInputs_twinwall();

    std::cout << "\nAll twin-wall NN tests (fractional) passed ✅\n";
    return 0;
}
