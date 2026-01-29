# Eigenfaces API Reference

## Overview
This document describes all public APIs and functions available in the Eigenfaces project for face recognition using Principal Component Analysis (PCA).

---

## Core Functions

### 1. `vector<Mat> read_faces()`
**Location:** Line 96  
**Purpose:** Reads face images from disk into memory

**Signature:**
```cpp
vector<Mat> read_faces()
```

**Returns:**
- `vector<Mat>`: Vector containing all loaded face images (19×19 pixels each)
- Total images loaded: `NO_OF_IMAGES` (2429 by default)

**Description:**
- Loads face images from `images/train/face` directory
- Image format: `.pgm` files
- Each image is loaded as 8-bit unsigned single-channel (`CV_8UC1`)
- Images are named as `faceXXXXX.pgm` where XXXXX is zero-padded index

**Example Usage:**
```cpp
vector<Mat> faces = read_faces();
cout << "Loaded " << faces.size() << " face images" << endl;
```

---

### 2. `vector<Mat> extract_train_test_set(vector<Mat> faces, vector<Mat> &test_set)`
**Location:** Line 119  
**Purpose:** Splits face images into 90/10 training and test sets

**Signature:**
```cpp
vector<Mat> extract_train_test_set(
    vector<Mat> faces,           // Input: all face images
    vector<Mat> &test_set        // Output: test images (pass by reference)
)
```

**Parameters:**
- `faces`: Vector of all loaded face images
- `test_set`: Reference to vector that will hold test images (10%)

**Returns:**
- `vector<Mat>`: Training set (90% of images, ~2186 images)

**Description:**
- Splits input faces into training (first 90%) and test (last 10%) sets
- Training set: 2186 images
- Test set: 243 images
- No randomization applied (uses sequential split)

**Example Usage:**
```cpp
vector<Mat> training_set, test_set;
training_set = extract_train_test_set(faces, test_set);
cout << "Training: " << training_set.size() << ", Test: " << test_set.size() << endl;
```

---

### 3. `void minDistComputation(map<int, vector<double>>::iterator iter, vector<Point>& kNeighbours, int k, Point& minLoc, Point& maxLoc)`
**Location:** Line 158  
**Purpose:** Computes minimum distance for k-nearest neighbors

**Signature:**
```cpp
void minDistComputation(
    map<int, vector<double>>::iterator iter,  // Iterator to distance map
    vector<Point>& kNeighbours,               // Output: k nearest neighbors
    int k,                                     // Number of neighbors to find
    Point& minLoc,                             // Output: location of minimum
    Point& maxLoc                              // Output: location of maximum
)
```

**Parameters:**
- `iter`: Iterator pointing to map entry with distance vector
- `kNeighbours`: Reference to vector storing neighbor indices
- `k`: Number of nearest neighbors
- `minLoc`, `maxLoc`: Output locations of min/max distances

**Returns:**
- `void` (modifies `kNeighbours`, `minLoc`, `maxLoc` by reference)

**Description:**
- Uses OpenCV's `minMaxLoc()` to find minimum distance in a set
- Adds minimum location to kNeighbours vector k times
- Useful for k-NN classification

---

### 4. `void calcNearestNeighbours(Mat& XTest, Mat& XTrain, vector<Mat>& nearestNeighbour, vector<int> testIndices, int training_examples_count, int kNeighbours, string space)`
**Location:** Line 174  
**Purpose:** Calculates nearest neighbors in original or PCA subspace

**Signature:**
```cpp
void calcNearestNeighbours(
    Mat& XTest,                      // Test data matrix (361 × n_test)
    Mat& XTrain,                     // Training data matrix (361 × n_train)
    vector<Mat>& nearestNeighbour,   // Output: nearest neighbor matrices
    vector<int> testIndices,         // Indices of test samples to process
    int training_examples_count,     // Number of training examples (2186)
    int kNeighbours,                 // k value for k-NN (typically 1)
    string space                     // Space description ("Original Space" or "Sub-space")
)
```

**Parameters:**
- `XTest`: Test data matrix (361-dimensional feature vectors)
- `XTrain`: Training data matrix (361-dimensional feature vectors)
- `testIndices`: Indices of test samples to evaluate
- `training_examples_count`: Total number of training samples
- `kNeighbours`: k for k-NN algorithm
- `space`: String describing space ("Original Space" or "Sub-space")

**Returns:**
- `void` (outputs nearest neighbors and prints to console)

**Description:**
- Computes Euclidean distances between test and training samples
- Uses `norm(col2, col1)` for distance computation
- Prints nearest neighbor index for each test sample
- Can operate in both original feature space and PCA subspace

**Example Usage:**
```cpp
vector<Mat> nearestNeighbours;
calcNearestNeighbours(XTest_trans, XTrain_trans, nearestNeighbours, 
                      testSamplesIndices, training_set.size(), 1, "Original Space");
```

---

### 5. `void visualizeEigenVectors(Mat viz, const Mat& V, ofstream& file_eigen_vector_first_column, int index, vector<Mat>& eigenVectorViz)`
**Location:** Line 222  
**Purpose:** Visualizes eigenvectors as 19×19 face-like images

**Signature:**
```cpp
void visualizeEigenVectors(
    Mat viz,                                    // Input: row from eigenvector matrix
    const Mat& V,                               // Eigenvector matrix (361 × 361)
    ofstream& file_eigen_vector_first_column,   // Output file handle
    int index,                                  // Index of eigenvector to visualize
    vector<Mat>& eigenVectorViz                 // Output: vector of visualized eigenvectors
)
```

**Parameters:**
- `viz`: Working matrix for eigenvector row (19×19)
- `V`: Complete eigenvector matrix from eigendecomposition
- `file_eigen_vector_first_column`: File stream for debugging output
- `index`: Which eigenvector row to visualize (0-based)
- `eigenVectorViz`: Reference to accumulate visualized eigenvectors

**Returns:**
- `void` (modifies `eigenVectorViz` and writes to file)

**Description:**
- Extracts eigenvector row from matrix V
- Reshapes 361-element vector into 19×19 matrix
- Normalizes values to 0-255 range
- Resizes to 200×200 pixels for visualization
- Saves as PNG image: `eigenVecViz{index}.png`

**Example Usage:**
```cpp
vector<Mat> eigenVectorViz;
Mat viz(19, 19, CV_32FC1);
for(int i = 0; i < 20; i++) {
    visualizeEigenVectors(viz, V, file_handle, i, eigenVectorViz);
}
```

---

### 6. `void plotDistances(QCustomPlot &customPlot1, const QVector<double>& column1, QVector<double>& column2, QMainWindow &window1, int plotNumber)`
**Location:** Line 265  
**Purpose:** Creates and displays distance distribution plots

**Signature:**
```cpp
void plotDistances(
    QCustomPlot &customPlot1,       // QCustomPlot widget for rendering
    const QVector<double>& column1, // X-axis data (training image indices)
    QVector<double>& column2,       // Y-axis data (distances)
    QMainWindow &window1,           // Parent window
    int plotNumber                  // Plot identifier (for naming)
)
```

**Parameters:**
- `customPlot1`: QCustomPlot graph widget
- `column1`: Training image indices (0 to 2186)
- `column2`: Euclidean distances from test sample
- `window1`: Main window containing the plot
- `plotNumber`: Sequential plot number (0-9)

**Returns:**
- `void` (displays plot and saves as JPEG)

**Description:**
- Creates line graph of distance distribution
- Sets axis labels and ranges
- Enables interactive zooming and dragging
- Saves plot as JPEG: `{plotNumber}{maxDistance}.png`
- Window title shows plot number and type

**Features:**
- Interactive range dragging (`iRangeDrag`)
- Interactive zooming (`iRangeZoom`)
- Antialiasing disabled for performance
- Window size: 500×400 pixels

---

## Main Function

### `int main(int argc, char **argv)`
**Location:** Line 308  
**Purpose:** Main entry point - orchestrates entire eigenface computation pipeline

**Parameters:**
- `argc`, `argv`: Standard command-line arguments (not used currently)

**Returns:**
- `0`: Successful execution
- `1`: Error (eigenvalue threshold not met)

**Workflow:**
1. Load and shuffle face images
2. Split into 90/10 training/test sets
3. Construct feature matrices (361-dimensional vectors)
4. Compute mean face
5. Center training data
6. Compute covariance matrix
7. Compute eigenvalues and eigenvectors
8. Find k eigenvectors that explain 90% of variance (typically k=20)
9. Visualize top k eigenvectors as face-like images
10. Project test images into PCA subspace
11. Compute distances in original and subspace
12. Find nearest neighbors in both spaces
13. Generate and save plots
14. Clean up resources

---

## Constants and Global Variables

### Macro Definition
```cpp
#define NO_OF_IMAGES 2429  // Total number of face images
```

### Global Static Variables
```cpp
static int colSize = 0;     // Column index for matrix construction
static int matIndex;        // Matrix index for visualization
```

---

## Data Types and Structures

### Matrix Dimensions
| Variable | Rows | Cols | Type | Purpose |
|----------|------|------|------|---------|
| `X_train1` | 2186 | 361 | CV_8UC1 | Training data (raw) |
| `XTrain_trans` | 361 | 2186 | CV_32FC1 | Transposed training data |
| `covariance_matrix` | 361 | 361 | CV_32FC1 | Covariance matrix |
| `E` | 1 | 361 | CV_32FC1 | Eigenvalues (sorted) |
| `V` | 361 | 361 | CV_32FC1 | Eigenvectors (rows) |
| `projectionMatrix_trainingData` | 20 | 2186 | CV_32FC1 | Training projections |
| `projectionMatrix_testData` | 20 | 243 | CV_32FC1 | Test projections |

---

## File I/O Operations

### Output Files Generated

| Path | Content |
|------|---------|
| `images/train/X_center.dat` | Centered training matrix |
| `images/train/X_covariance.dat` | Covariance matrix |
| `images/train/X_eigen_values.dat` | Eigenvalues |
| `images/train/X_eigen_vector_first_col.dat` | Eigenvector metadata |
| `images/train/eigenVecViz/*.png` | Eigenvector visualizations |
| `images/coVarMat.dat` | Alternative covariance matrix |
| `images/Spectrum-of-co-variance.png` | Eigenvalue spectrum plot |
| `images/projection_matrix_trainingData.dat` | Training PCA projections |
| `images/projection_matrix_testData.dat` | Test PCA projections |
| `images/EuclideandistancesPlots/*.png` | Distance plots |
| `images/test/X_test.dat` | Test data matrix |
| `images/test/X_test_train_distance.dat` | Original space distances |
| `images/test/X_test_train_distance_after_pca.dat` | PCA subspace distances |

---

## OpenCV API Dependencies

### Functions Used
- `imread()`: Load images
- `Mat::t()`: Matrix transpose
- `Mat::clone()`: Deep copy matrix
- `Mat::convertTo()`: Type conversion
- `divide()`: Element-wise division
- `norm()`: Euclidean distance
- `eigen()`: Eigendecomposition
- `calcCovarMatrix()`: Covariance computation
- `minMaxLoc()`: Find min/max locations
- `normalize()`: Normalize matrix values
- `imwrite()`: Save images
- `imshow()`: Display images

---

## Qt API Dependencies

### Classes/Methods Used
- `QApplication`: Application instance
- `QMainWindow`: Window container
- `QCustomPlot`: 2D plotting widget
- `QCPPlotTitle`: Plot title
- `QVector<double>`: Dynamic vector
- `QString`: String class
- `QFile`: File operations

---

## Usage Example

```cpp
// In main():
1. Load all faces
   vector<Mat> faces = read_faces();

2. Split data
   vector<Mat> training_set, test_set;
   training_set = extract_train_test_set(faces, test_set);

3. Construct matrices and compute PCA
   Mat X_train1(2186, 361, CV_8UC1);
   // ... fill matrix ...
   Mat covariance_matrix = X_train_trans * X_train_trans.t();
   
4. Get eigenvectors
   Mat E, V;
   eigen(covariance_matrix, E, V);

5. Find k eigenvectors
   Mat smallestKEigenVectors = V.rowRange(0, 20);

6. Project to PCA space
   Mat projectionMatrix = smallestKEigenVectors * XTrain_trans;

7. Find nearest neighbors
   calcNearestNeighbours(XTest_trans, XTrain_trans, 
                        nearestNeighbours, testIndices, 
                        training_set.size(), 1, "Original Space");
```

---

## Compilation

```bash
cd /Users/friedrichhahn/Projekte/Eigenfaces
mkdir build && cd build
cmake ..
make
```

**Dependencies:**
- OpenCV 4.13.0+
- Qt 6.10.1+
- QCustomPlot library
- CMake 3.10+

---

## Notes

- All matrices use OpenCV's `Mat` class
- Features are 361-dimensional (19×19 image flattened)
- PCA reduces to k=20 dimensions to capture ~90% variance
- k-NN classification uses k=1 (nearest single neighbor)
- All pixel values are in range [0, 255]
