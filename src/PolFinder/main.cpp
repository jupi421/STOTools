#include <polfinder.hpp>
#include <Eigen/Core>
#include <vector>

int main() {
	const char* name = "./POSCAR";
	std::vector<Eigen::Vector3d> test = PolFinder::loadPosFromFile(name, 8);
	PolFinder::AtomPositions test_sorted = PolFinder::sortPositions(test, 96, 96, 288);

}
