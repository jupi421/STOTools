#include <polfinder.hpp>
#include <Eigen/Core>
#include <vector>


const char* name = "./POSCAR";

void testFileReader() {
	PolFinder::Positions test = PolFinder::loadPosFromFile(name, 8, 105);
	for(auto &i : test) {
		std::printf("  %.16f  %.16f  %.16f\n", i[0], i[1], i[2]);
	}
	assert(test.size() == 96);
}

void testSortPositions() {
	PolFinder::Positions positions = PolFinder::loadPosFromFile(name, 8);

	PolFinder::AtomPositions sorted_positions = PolFinder::sortPositionsByType(positions, 96, 96, 288);
	PolFinder::Positions Sr_positions = PolFinder::loadPosFromFile(name, 8, 105);
	PolFinder::Positions Ti_positions = PolFinder::loadPosFromFile(name, 104, 105+96);
	PolFinder::Positions O_positions = PolFinder::loadPosFromFile(name, 104+96);

	for(size_t i { 0 }; i < Sr_positions.size(); i++){
		assert(sorted_positions.SrPositions[i][0] == Sr_positions[i][0]);
		assert(sorted_positions.SrPositions[i][1] == Sr_positions[i][1]);
		assert(sorted_positions.SrPositions[i][2] == Sr_positions[i][2]);

		assert(sorted_positions.TiPositions[i][0] == Ti_positions[i][0]);
		assert(sorted_positions.TiPositions[i][1] == Ti_positions[i][1]);
		assert(sorted_positions.TiPositions[i][2] == Ti_positions[i][2]);

		assert(sorted_positions.OPositions[i][0] == O_positions[i][0]);
		assert(sorted_positions.OPositions[i][1] == O_positions[i][1]);
		assert(sorted_positions.OPositions[i][2] == O_positions[i][2]);
	}
}

void testGetNearestNeighbors();

int main() {
	testFileReader();
	testSortPositions();

	std::cout << "All tests completed!" << std::endl;
} 
