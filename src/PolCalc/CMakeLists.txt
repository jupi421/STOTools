project(PolCalc)

set(CMAKE_CXX_STANDARD 23)
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

#pybind11_add_module(polcalc main.cpp) 
add_library(polcalc_lib INTERFACE)
target_include_directories(polcalc_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(polcalc_lib INTERFACE Eigen3::Eigen)

add_executable(test tests.cpp)
target_link_libraries(test PRIVATE polcalc_lib)
