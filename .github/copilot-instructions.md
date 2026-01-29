# Copilot Instructions for Eigenfaces Project

## Project Overview
Eigenfaces is a C++ face recognition system using Principal Component Analysis (PCA). It processes 2429 training images (19×19 pixels each), performs PCA dimensionality reduction, and performs k-NN classification on test data.

## Architecture & Data Flow

### Key Data Processing Pipeline
1. **Load Images**: `read_faces()` reads 2429 `.pgm` face images from `images/train/face/faceXXXXX.pgm`
2. **Split Data**: `extract_train_test_set()` creates 90/10 train/test split (2186 train, 243 test)
3. **Matrix Construction**: Flatten 19×19 images into 361-dimensional vectors, stack into matrices
4. **PCA Transformation**:
   - Compute mean image and center data
   - Calculate covariance matrix using `cv::calcCovarMatrix()`
   - Decompose via `SVD` (eigenvalues/eigenvectors)
   - Select top K eigenvectors for projection space
5. **Classification**: Project test samples, compute Euclidean distances to training data, find k-NN

### Critical Dependencies
- **OpenCV 4.x**: Image I/O, matrix operations, PCA/SVD
  - Note: Use `COVAR_ROWS` (not deprecated `CV_COVAR_ROWS`)
  - Covariance computation done on row-centered data
- **GTest**: Unit testing framework
- **Qt6** (optional): GUI components (QCustomPlot) - currently disabled in CMake

## Code Patterns & Conventions

### Image & Matrix Handling
- **Format**: 19×19 grayscale (CV_8UC1) images loaded as individual `cv::Mat` objects
- **Vectorization**: Images flattened to 361-dimensional vectors, stored as matrix rows
- **Type Safety**: Always cast `.size()` to `(int)` in loops to avoid signed/unsigned warnings:
  ```cpp
  for (int i = 0; i < (int)training_set.size(); i++) { }
  ```

### Fixed Data Sizes
- `NO_OF_IMAGES = 2429` (constant, preprocessor define)
- Training size: 2186 (90%)
- Test size: 243 (10%)
- Image dimension: 361 (19×19)

### Distance Computation
- Uses OpenCV's `cv::norm(col1, col2, NORM_L2)` for Euclidean distance
- Returns `Scalar` object; access value via `.val[0]`

### File I/O Pattern
- Output matrices/distances to `.dat` files in `images/test/` directory
- Example: `X_test_train_distance_after_pca.dat`
- **Important**: Always close file streams after writing (memory leak fix applied)

## Build & Test Workflow

### Build
```bash
cd build
cmake ..
make
```

### Run Tests
```bash
cd build
./test_eigenfaces
# Or via CMake
ctest
```

### Test Framework
- Uses **GTest** with test fixtures (see [test_eigenfaces.cpp](test_eigenfaces.cpp))
- Tests verify: image dimensions (19×19 = 361 elements), train/test split proportions, matrix construction
- No actual face images needed for unit tests (uses synthetic images)

## Critical Fixes & Known Issues

### Memory & Type Safety
1. **File Handles**: Ensure all `ofstream`/`ifstream` objects are closed (lines 763-769)
2. **Uninitialized Variables**: `sumOfEigenValues` must be initialized to `0.0f`
3. **Size Comparisons**: Cast `.size()` to `(int)` in all loop conditions

### OpenCV Compatibility
- Use `COVAR_ROWS` instead of deprecated `CV_COVAR_ROWS` (line 483)
- Verify with OpenCV 4.13.0

### GUI Components
- QCustomPlot dependency is not installed; main executable is commented out in CMakeLists.txt
- Test executable (without GUI) builds successfully

## File Reference Guide

| File | Purpose |
|------|---------|
| [eigenfaces.cpp](eigenfaces.cpp) | Main implementation (506 lines) |
| [test_eigenfaces.cpp](test_eigenfaces.cpp) | GTest test suite (430 lines) |
| [CMakeLists.txt](CMakeLists.txt) | Build configuration |
| [API_REFERENCE.md](API_REFERENCE.md) | Detailed function signatures |
| [FIXES_APPLIED.md](FIXES_APPLIED.md) | Bug fixes log |

## Common Development Tasks

**Adding a new distance metric**: Modify loop in `calcNearestNeighbours()` or `main()` where `cv::norm()` is called.

**Changing train/test split**: Update `percentage_train` calculation in `extract_train_test_set()` (line 42) and test expectations.

**Tuning PCA dimensionality**: Modify eigenvalue threshold selection logic (around line 548) to keep different number of components.

**Debugging classification**: Output `projectionMatrix_testData` and `projectionMatrix_trainingData` to `.dat` files for analysis.
