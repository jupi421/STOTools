#include "polfinder.hpp"
#include <Eigen/Core>
#include <vector>

using namespace PolFinder;
int main() {
	const char* name = "./POSCAR";
	std::vector<Eigen::Vector3d> test = loadPosFromFile(name, 8);
	AtomPositions test_sorted = sortPositions(test, 96, 96, 288);
	std::vector<NearestNeighbors> test_NN = getNearestNeighbors(test_sorted, 5);
}
