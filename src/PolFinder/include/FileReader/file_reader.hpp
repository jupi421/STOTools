# pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Core>

namespace FileReader {
	inline void loadPosFromFile(std::string filename, uint head) {
		std::ifstream file { filename } ;
		
		if (!file.is_open()) {
			throw std::runtime_error("Failed loading file!");
		}

		std::string line;
		uint skip { head };
		
		Eigen::Vector3d position;
		std::vector<Eigen::Vector3d> positions;
		while(std::getline(file, line)) {
			if (skip) {
				--skip;
				continue;
			}
			
			line = line.substr(2, line.length());
			std::string pos_x = line.substr(0, line.find("  "));
			std::string pos_y = line.substr(pos_x.length() + 2, line.find("  "));
			std::string pos_z = line.substr(pos_x.length()+pos_y.length()+4, line.find(" "));
			
			// TODO Split into atom species
			position = { std::stod(pos_x), std::stod(pos_y), std::stod(pos_z) };
			positions.push_back(position);
		}
		
		file.close();
	};
}
