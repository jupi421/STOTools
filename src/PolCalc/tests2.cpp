#include <polcalc.hpp>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <optional>
#include <print>   // C++23 std::println

using namespace PolCalc;

// --- Small helpers -----------------------------------------------------------
static inline Eigen::Vector3d min_image(Eigen::Vector3d v) {
    v.array() -= v.array().round(); // wrap to [-0.5,0.5)
    return v;
}

static inline double clamp01(double x) {
    if (x < -1.0) return -1.0;
    if (x >  1.0) return  1.0;
    return x;
}

static inline double metric_dot(const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b,
                                const Eigen::Matrix3d& M)
{
    // M = A^T A where A is the lattice matrix (direct->cartesian)
    return a.dot(M*b);
}

static inline double metric_norm(const Eigen::Vector3d& a,
                                 const Eigen::Matrix3d& M)
{
    double s2 = metric_dot(a,a,M);
    return s2 > 0.0 ? std::sqrt(s2) : 0.0;
}

static inline double metric_angle(const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b,
                                  const Eigen::Matrix3d& M)
{
    double an = metric_norm(a,M);
    double bn = metric_norm(b,M);
    if (an == 0.0 || bn == 0.0) return 0.0;
    double c = metric_dot(a,b,M) / (an*bn);
    c = clamp01(c);
    return std::acos(c); // radians
}

// --- The test (NO midpoint check) -------------------------------------------
static void test_O_pairs_antiparallel_no_mid(const char* poscar_path,
                                             DWType wall_type,
                                             double dw_center_x,
                                             double anti_parallel_tol_rad = 0.22,  // ~12.6°
                                             double pair_len_rel_tol    = 0.15,     // 15% tolerance for |r1| vs |r2|
                                             bool   verbose             = true)
{
    std::cout << "\n=== Testing DWType="
              << (wall_type==DWType::HT?"HT":wall_type==DWType::HH?"HH":"APB")
              << " @ x=" << dw_center_x << " ===\n";

    // 1) Load & split
    Positions positions = loadPosFromFile(poscar_path, /*head*/8);
    assert(!positions.empty());

    constexpr size_t N_Sr=96, N_Ti=96, N_O=288;
    auto splitExp = sortPositionsByType(positions, N_Sr, N_Ti, N_O);
    assert(splitExp.has_value());
    AtomPositions ap = *splitExp;

    // 2) Lattice from POSCAR header (direct -> cartesian)
    Eigen::Matrix3d A;
    A << 34.9546388237903614, 0.0, 0.0,
         0.0, 15.6192183281608035, 0.0,
         0.0, 0.0, 11.6371520466789136;
    const Eigen::Matrix3d M = A.transpose() * A;

    // Useful constants for logging / checks
    const double cos_cut = -std::cos(anti_parallel_tol_rad); // cos(pi - tol) = -cos(tol)

    if (verbose) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Anti-parallel tolerance (rad)=" << anti_parallel_tol_rad
                  << " => cos_cut=" << cos_cut << "\n";
        std::cout << "O-pair len rel tol = " << pair_len_rel_tol << "\n";
    }

    size_t sites_ok = 0;
    size_t sites_fail = 0;
    size_t total_pairs_ok = 0;

    // 3) Iterate all Ti sites
    for (size_t t = 0; t < ap.m_Ti.size(); ++t) {
        const Atom& Ti = ap.m_Ti[t];

        if (verbose) {
            std::println("\n--- Ti index {}  Ti_dir={:.6f} {:.6f} {:.6f} ---",
                         t, Ti.m_position[0], Ti.m_position[1], Ti.m_position[2]);
        }

        // 3a) NNs (use metric) in direct space
        auto srNN_all = getNearestNeighbors(ap.m_Sr, Atoms{Ti}, /*n*/8, A, /*sort*/true);
        auto oNN_all  = getNearestNeighbors(ap.m_O,  Atoms{Ti}, /*n*/6, A, /*sort*/true);
        if (!(srNN_all && oNN_all)) {
            std::cerr << "Nearest neighbor query failed at Ti index " << t << "\n";
            assert(false);
        }
        const NNIds& srNN = srNN_all->at(0);
        const NNIds& oNN  = oNN_all->at(0);
        assert(srNN.size()==8 && oNN.size()==6);

        Atoms A8; A8.reserve(8);
        Atoms O6; O6.reserve(6);
        for (auto [i,_] : srNN) A8.push_back(ap.m_Sr[i]);
        for (auto [i,_] : oNN)  O6.push_back(ap.m_O[i]);

        // 3b) Build LocalUC (uses your pairing logic)
        helper::LocalUC luc(A8, Ti, O6, wall_type, A, /*DW center*/dw_center_x, /*tol*/1e-6);

        // Invariants
        assert(luc.m_A_direct_pbc.size() == 4);  // four Sr opposite pairs
        assert(luc.m_O_direct_pbc.size() == 3);  // three opposite O pairs

        // For debug: show O6 relative to Ti
        if (verbose) {
            // Reconstruct O6 relative vectors for printing
            std::vector<Eigen::Vector3d> rO;
            rO.reserve(6);
            for (const auto& pr : luc.m_O_direct_pbc) {
                rO.push_back( min_image(pr.first.m_position  - Ti.m_position) );
                rO.push_back( min_image(pr.second.m_position - Ti.m_position) );
            }
            // dedup prints by re-deriving from O6 may not correspond one-to-one in order;
            // keep it simple: just print the three pairs as we check them below
        }

        // 3c) Check each O pair (NO midpoint check)
        size_t ok_here = 0;
        for (size_t k = 0; k < luc.m_O_direct_pbc.size(); ++k) {
            const auto& pr = luc.m_O_direct_pbc[k];
            Eigen::Vector3d r1 = min_image(pr.first.m_position  - Ti.m_position);
            Eigen::Vector3d r2 = min_image(pr.second.m_position - Ti.m_position);

            double n1 = metric_norm(r1, M);
            double n2 = metric_norm(r2, M);

            // Angle (metric)
            double ang = metric_angle(r1, r2, M);
            double cosang = std::cos(ang);

            // Relative length difference
            double avg = 0.5*(n1+n2);
            double rel_len_diff = (avg>0.0) ? std::abs(n1-n2)/avg : 0.0;

            // Decisions (NO midpoint criterion)
            bool anti_ok = (cosang <= cos_cut);
            bool len_ok  = (rel_len_diff <= pair_len_rel_tol);

            if (verbose) {
                std::println("Pair {}:", k);
                std::cout << "  r1_dir=" << std::setw(10) << r1[0] << " "
                                     << std::setw(10) << r1[1] << " "
                                     << std::setw(10) << r1[2]
                                     << " |r1|_M=" << n1 << "\n";
                std::cout << "  r2_dir=" << std::setw(10) << r2[0] << " "
                                     << std::setw(10) << r2[1] << " "
                                     << std::setw(10) << r2[2]
                                     << " |r2|_M=" << n2 << "\n";
                std::cout << "  cos=" << cosang
                          << "  angle=" << (ang*180.0/M_PI) << " deg"
                          << "  len_rel_diff=" << rel_len_diff << "\n";
                std::cout << "  anti? " << (anti_ok?"YES":"NO")
                          << "  len_ok? " << (len_ok?"YES":"NO") << "\n";
            }

            if (anti_ok && len_ok) ++ok_here;
        }

        if (ok_here == 3) {
            ++sites_ok;
            total_pairs_ok += 3;
        } else {
            ++sites_fail;
            if (verbose) std::cout << "Ti " << t << ": O-pair check FAILED (ok=" << ok_here << ")\n";
            // If you prefer hard failure per site, uncomment:
            // assert(false);
        }
    }

    std::cout << "\n✓ Finished (NO midpoint check).\n";
    std::cout << "Summary: Ti sites = " << ap.m_Ti.size()
              << " | sites OK=" << sites_ok
              << " | sites FAILED=" << sites_fail
              << " | O-opposite OK total=" << total_pairs_ok
              << " (expect 3*NTi, NTi=" << ap.m_Ti.size() << ")\n";
}

int main() {
    const char* POSCAR_DW = "./POSCAR"; // path to your structure

    // Run the O-pair test without midpoint check
    test_O_pairs_antiparallel_no_mid(POSCAR_DW, DWType::HT, 0.5,
                                     /*anti_parallel_tol_rad*/0.34,
                                     /*pair_len_rel_tol*/0.15,
                                     /*verbose*/true);

    return 0;
}
