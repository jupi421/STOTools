#include <polcalc.hpp>
#include <Eigen/Core>
#include <vector>

const char* name = "./POSCAR";

void testFileReader() {
	PolCalc::Positions test = PolCalc::loadPosFromFile(name, 8, 105);
	for(auto &i : test) {
		//std::printf("  %.16f  %.16f  %.16f\n", i[0], i[1], i[2]);
	}
	assert(test.size() == 96);
}

void testSortPositions() {
	PolCalc::Positions positions = PolCalc::loadPosFromFile(name, 8);

	//PolCalc::AtomPositions sorted_positions = PolCalc::sortPositionsByType(positions, 96000, 96000, 288000);
	//PolCalc::Positions Sr_positions = PolCalc::loadPosFromFile(name, 8, 96000+8+1);
	//PolCalc::Positions Ti_positions = PolCalc::loadPosFromFile(name, 96008, 96008+8+1 +96000);
	//PolCalc::Positions O_positions = PolCalc::loadPosFromFile(name, 96008+8+96000+96);

	//PolCalc::AtomPositions sorted_positions = PolCalc::sortPositionsByType(positions, 96, 96, 288);
	//PolCalc::Positions Sr_positions = PolCalc::loadPosFromFile(name, 8, 105);
	//PolCalc::Positions Ti_positions = PolCalc::loadPosFromFile(name, 104, 105+96);
	//PolCalc::Positions O_positions = PolCalc::loadPosFromFile(name, 104+96);

	//for(size_t i { 0 }; i < Sr_positions.size(); i++){
	//	assert(sorted_positions.SrPositions[i][0] == Sr_positions[i][0]);
	//	assert(sorted_positions.SrPositions[i][1] == Sr_positions[i][1]);
	//	assert(sorted_positions.SrPositions[i][2] == Sr_positions[i][2]);

	//	assert(sorted_positions.TiPositions[i][0] == Ti_positions[i][0]);
	//	assert(sorted_positions.TiPositions[i][1] == Ti_positions[i][1]);
	//	assert(sorted_positions.TiPositions[i][2] == Ti_positions[i][2]);

	//	assert(sorted_positions.OPositions[i][0] == O_positions[i][0]);
	//	assert(sorted_positions.OPositions[i][1] == O_positions[i][1]);
	//	assert(sorted_positions.OPositions[i][2] == O_positions[i][2]);

	//}
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

	PolCalc::Positions positions = PolCalc::loadPosFromFile(name, 8);
	PolCalc::AtomPositions sorted_positions = PolCalc::sortPositionsByType(positions, 96, 96, 288);
	std::vector<PolCalc::NearestNeighborsByType> nearest_neighbors = PolCalc::getNearestNeighborsAll(sorted_positions, 8, cell_matrix);
	
	PolCalc::NearestNeighborsByType Sr_NN { nearest_neighbors.at(0) }; 
	PolCalc::NearestNeighborsByType Ti_NN { nearest_neighbors.at(1) };
	PolCalc::NearestNeighborsByType O_NN { nearest_neighbors.at(2) };

	std::cout << '\n';
	for (size_t i { 96 }; i < 2*96; i++) {
		std::cout << (positions.at(i) == sorted_positions.TiPositions.at(i-96)) << std::endl;
	}
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
		std::vector<std::pair<size_t, double>> reference_atom_NN { Sr_NN.Ti_NN_ids.at(reference_atom_id) }; 

		for(const std::pair<size_t, double> &NN_ids : reference_atom_NN) {
			std::cout << "Ref: " << reference_atom_id << "  " << "Cur: " << (NN_ids.first + 96) << "  " << "dist: " << NN_ids.second << std::endl;
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
