# Eigenfaces Execution Guide

This document explains how to build and run the Eigenfaces project and how to execute the visualizer with the training data.

## Prerequisites

- CMake
- A C++ compiler supporting C++11 or newer
- OpenCV 4.x
- Qt 6 (for `visualize_eigenfaces_qt`)

## Build Instructions

From the project root:

```bash
cd /Users/friedrichhahn/Projekte/Eigenfaces
cmake -S . -B build
cmake --build build
```

This generates the built executables in the `build/` directory.

## Run Unit Tests

To verify the code, run:

```bash
cd /Users/friedrichhahn/Projekte/Eigenfaces
./build/test_eigenfaces
```

Expected output: `29 tests` passed.

## Run the Qt Visualizer

The Qt visualizer must be run from the project root so that the relative image path resolves correctly.

```bash
cd /Users/friedrichhahn/Projekte/Eigenfaces
./build/visualize_eigenfaces_qt images/train/face
```

### Notes

- If you launch from `build/` using `../images/train/face`, the application may fail to find images and abort.
- The visualizer saves generated plots to `images/train/eigenVecVis/`.

## Run the Non-GUI Visualizer

If available, the non-GUI version can be run similarly:

```bash
cd /Users/friedrichhahn/Projekte/Eigenfaces
./build/visualize_eigenfaces images/train/face
```

## Output

The visualizer creates:

- Mean face image
- Eigenface grid
- Eigenvalue spectrum plot
- Cumulative variance plot

Saved output is written into `images/train/eigenVecVis/` by default.

## Troubleshooting

- Ensure the `images/train/face` directory exists and contains the expected `.pgm` files.
- Use the project root as the current working directory when running the executable.
- If the program prints OpenCV warnings like `can't open/read file`, verify the path and file permissions.
