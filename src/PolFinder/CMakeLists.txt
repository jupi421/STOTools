project(PolFinder)

include(FetchContent)
FetchContent_declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

FetchContent_declare(
	Eigen
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#pybind11_add_module(polfinder main.cpp) 
add_library(file_reader_lib INTERFACE)
target_include_directories(file_reader_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/FileReader)
target_link_libraries(file_reader_lib INTERFACE Eigen3::Eigen)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE file_reader_lib)
