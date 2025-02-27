#include <file_reader.hpp>
#include <Eigen/Core>

int main() {
	const char* name = "./POSCAR";
	FileReader::loadPosFromFile(name, 8);
}
