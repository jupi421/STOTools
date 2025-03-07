project(PolFinder)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

#pybind11_add_module(polfinder main.cpp) 
add_library(polfinder_lib INTERFACE)
target_include_directories(polfinder_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(polfinder_lib INTERFACE Eigen3::Eigen)

add_executable(test tests.cpp)
target_link_libraries(test PRIVATE polfinder_lib)
