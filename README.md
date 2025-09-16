## Polarization Calculation tool for STO
### Requirements
- CMake (3.31.5)
- Ninja

### Installation
```
git clone https://github.com/jupi421/STOTools.git
cd STOTools
cmake -S . -B build -G "Ninja Multi-Config"
cmake --build build --config Release 
```
Use "Debug" instead of "Release" to include debug symbols and optimization O0.
