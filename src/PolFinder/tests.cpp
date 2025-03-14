#include <polfinder.hpp>
#include <Eigen/Core>
#include <vector>

const char* name = "./POSCAR";

void testFileReader() {
	PolFinder::Positions test = PolFinder::loadPosFromFile(name, 8, 105);
	for(auto &i : test) {
		//std::printf("  %.16f  %.16f  %.16f\n", i[0], i[1], i[2]);
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
	cell_matrix << 34.9546388237903614,   0,	   0,
					0,		 15.6192183281608035,  0,
					0,		  0,	  11.6371520466789136;

	
	//cell_matrix << 19.72565,   0,	    0,
	//				0,		   7.89026, 0,
	//				0,		   0,	    7.89026;
	
	//cell_matrix << 19.72565,   0,	    0,
	//				0,		   19.72565, 0,
	//				0,		   0,	    19.72565;

	PolFinder::Positions positions = PolFinder::loadPosFromFile(name, 8);
	PolFinder::AtomPositions sorted_positions = PolFinder::sortPositionsByType(positions, 96, 96, 288);
	std::vector<PolFinder::NearestNeighbors> nearest_neighbors = PolFinder::getNearestNeighbors(sorted_positions, cell_matrix, 8);
	
	PolFinder::NearestNeighbors Sr_NN { nearest_neighbors.at(0) }; 
	PolFinder::NearestNeighbors Ti_NN { nearest_neighbors.at(1) };
	PolFinder::NearestNeighbors O_NN { nearest_neighbors.at(2) };

	//assert(Sr_NN.Sr_NN_ids.size() == 20);
	//assert(Sr_NN.Ti_NN_ids.size() == 20);
	//assert(Sr_NN.O_NN_ids.size() == 20);
	//

	//assert(Ti_NN.Sr_NN_ids.size() == 20);
	//assert(Ti_NN.Ti_NN_ids.size() == 20);
	//assert(Ti_NN.O_NN_ids.size() == 20);

	//assert(O_NN.Sr_NN_ids.size() == 60);
	//assert(O_NN.Ti_NN_ids.size() == 60);
	//assert(O_NN.O_NN_ids.size() == 60);
	//
	//// Sr NN
	//for(const auto &NN_list : Sr_NN.Sr_NN_ids) {
	//	assert(NN_list.size() == n);
	//}
	//for(const auto &NN_list : Sr_NN.Ti_NN_ids) {
	//	assert(NN_list.size() == n);
	//}
	//for(const auto &NN_list : Sr_NN.O_NN_ids) {
	//	assert(NN_list.size() == n);
	//}

	//// Ti NN
	//for(const auto &NN_list : Ti_NN.Sr_NN_ids) {
	//	assert(NN_list.size() == n);
	//}
	//for(const auto &NN_list : Ti_NN.Ti_NN_ids) {
	//	assert(NN_list.size() == n);
	//}
	//for(const auto &NN_list : Ti_NN.O_NN_ids) {
	//	assert(NN_list.size() == n);
	//}

	//// O NN
	//for(const auto &NN_list : O_NN.Sr_NN_ids) {
	//	assert(NN_list.size() == n);
	//}
	//for(const auto &NN_list : O_NN.Ti_NN_ids) {
	//	assert(NN_list.size() == n);
	//}
	//for(const auto &NN_list : O_NN.O_NN_ids) {
	//	assert(NN_list.size() == n);
	//}

	// TODO test distances are within range 
	double threshold { };
	for(size_t reference_atom_id { 0 }; reference_atom_id < Sr_NN.Sr_NN_ids.size(); reference_atom_id++) {
		Eigen::Vector3d reference_atom { sorted_positions.SrPositions.at(reference_atom_id) };
		std::vector<std::pair<size_t, double>> reference_atom_NN { Sr_NN.Sr_NN_ids.at(reference_atom_id) }; 

		//std::cout << reference_atom.transpose() << '\n';
		for(const std::pair<size_t, double> &NN_ids : reference_atom_NN) {
			Eigen::Vector3d current_atom { sorted_positions.SrPositions.at(NN_ids.first) };
			Eigen::Vector3d dr { current_atom - reference_atom };

			double dx { std::abs(dr[0]) };
			double dy { std::abs(dr[1]) };
			double dz { std::abs(dr[2]) };

			dx -= static_cast<int>(dx + 0.5);
			dy -= static_cast<int>(dy + 0.5);
			dz -= static_cast<int>(dz + 0.5);
			
			double dist  {(cell_matrix*Eigen::Vector3d(dx, dy, dz)).norm()};
			
			std::cout << "Ref: " << reference_atom_id << "  " << "Cur: " << NN_ids.first << "  " << "dist: " << dist << std::endl;
		}
		std::cout << '\n';
	}
}

int main() {
	//testFileReader();
	//testSortPositions_sto_bulk();
	testGetNearestNeighbors();

	//std::cout << "All tests completed!" << std::endl;
} 
