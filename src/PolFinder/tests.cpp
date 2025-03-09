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

void testGetNearestNeighbors() {
	uint n = 8;
	Eigen::Matrix3d cell_matrix;
	cell_matrix << 23.3031,   0,	   0,
					0,		 15.6192,  0,
					0,		  0,	  17.5004;

	PolFinder::Positions positions = PolFinder::loadPosFromFile(name, 8);
	PolFinder::AtomPositions sorted_positions = PolFinder::sortPositionsByType(positions, 96, 96, 288);
	std::vector<PolFinder::NearestNeighbors> nearest_neighbors = PolFinder::getNearestNeighbors(sorted_positions, cell_matrix, 8);
	
	PolFinder::NearestNeighbors Sr_NN { nearest_neighbors.at(0) }; 
	PolFinder::NearestNeighbors Ti_NN { nearest_neighbors.at(1) };
	PolFinder::NearestNeighbors O_NN { nearest_neighbors.at(2) };

	assert(Sr_NN.Sr_NN_ids.size() == 96);
	assert(Sr_NN.Ti_NN_ids.size() == 96);
	assert(Sr_NN.O_NN_ids.size() == 96);

	assert(Ti_NN.Sr_NN_ids.size() == 96);
	assert(Ti_NN.Ti_NN_ids.size() == 96);
	assert(Ti_NN.O_NN_ids.size() == 96);

	assert(O_NN.Sr_NN_ids.size() == 288);
	assert(O_NN.Ti_NN_ids.size() == 288);
	assert(O_NN.O_NN_ids.size() == 288);
	
	// Sr NN
	for(const auto &NN_list : Sr_NN.Sr_NN_ids) {
		assert(NN_list.size() == n);
	}
	for(const auto &NN_list : Sr_NN.Ti_NN_ids) {
		assert(NN_list.size() == n);
	}
	for(const auto &NN_list : Sr_NN.O_NN_ids) {
		assert(NN_list.size() == n);
	}

	// Ti NN
	for(const auto &NN_list : Ti_NN.Sr_NN_ids) {
		assert(NN_list.size() == n);
	}
	for(const auto &NN_list : Ti_NN.Ti_NN_ids) {
		assert(NN_list.size() == n);
	}
	for(const auto &NN_list : Ti_NN.O_NN_ids) {
		assert(NN_list.size() == n);
	}

	// O NN
	for(const auto &NN_list : O_NN.Sr_NN_ids) {
		assert(NN_list.size() == n);
	}
	for(const auto &NN_list : O_NN.Ti_NN_ids) {
		assert(NN_list.size() == n);
	}
	for(const auto &NN_list : O_NN.O_NN_ids) {
		assert(NN_list.size() == n);
	}

	// TODO test distances are within range 
}

int main() {
	testFileReader();
	testSortPositions();
	testGetNearestNeighbors();

	std::cout << "All tests completed!" << std::endl;
} 
