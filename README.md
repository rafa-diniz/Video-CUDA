# Video-CUDA

A video preprocessing pipeline written in CUDA kernels and C++. Implements cropping, brightness changes, normalization and resizing.

---

## Quick Start

### Requirements

* CUDA
* CMake
* G++

### Installing

**Compile**:
```bash
cmake -S . -B build -G Ninja
cmake --build build
```

**Run**:
```bash
./build/video_cuda
```

---

## License

GPLv3 - see LICENSE for details.
