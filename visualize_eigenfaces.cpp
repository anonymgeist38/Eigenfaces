#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include "opencv2/opencv.hpp"

#define NO_OF_IMAGES 2429

using namespace std;
using namespace cv;

vector<Mat> read_faces() {
    vector<Mat> training_images;
    training_images.reserve(NO_OF_IMAGES);
    string images_path = "images/train/face";
    for (int i = 0; i < NO_OF_IMAGES; i++) {
        Mat img = imread(cv::format("%s/face%05d.pgm", images_path.c_str(), i), 0);
        if (!img.empty()) {
            training_images.push_back(img);
        }
    }
    return training_images;
}

vector<Mat> extract_train_test_set(vector<Mat> faces, vector<Mat> &test_set) {
    int percentage_train = (0.9f * NO_OF_IMAGES);
    vector<Mat> training_set;
    training_set.reserve(percentage_train);
    test_set.reserve(NO_OF_IMAGES - percentage_train);

    for (int i = 0; i < percentage_train; i++) {
        if (i < (int)faces.size()) training_set.push_back(faces[i]);
    }

    for (int i = percentage_train; i < (int)faces.size(); i++) {
        test_set.push_back(faces[i]);
    }

    return training_set;
}

void visualizeEigenVector(Mat eigenvector, int index) {
    // Reshape eigenvector from 361-D to 19x19
    Mat eigenface = eigenvector.reshape(1, 19);
    
    // Normalize to 0-255 range for visualization
    Mat dst;
    normalize(eigenface, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    
    // Resize for better visualization
    resize(dst, dst, Size(200, 200));
    
    // Save as PNG
    string filename = "images/train/eigenVecVis/eigenface_" + to_string(index) + ".png";
    bool success = imwrite(filename, dst);
    if (success) {
        cout << "Saved eigenface " << index << " to " << filename << endl;
    } else {
        cerr << "Failed to save " << filename << endl;
    }
}

int main() {
    cout << "========================================" << endl;
    cout << "Eigenfaces Visualization" << endl;
    cout << "========================================" << endl << endl;

    // Read and prepare data
    cout << "[1] Loading face images..." << endl;
    vector<Mat> faces = read_faces();
    cout << "Loaded " << faces.size() << " images" << endl << endl;

    cout << "[2] Splitting data..." << endl;
    vector<Mat> test_set;
    vector<Mat> training_set = extract_train_test_set(faces, test_set);
    cout << "Training set: " << training_set.size() << endl;
    cout << "Test set: " << test_set.size() << endl << endl;

    // Construct training matrix
    cout << "[3] Constructing training matrix..." << endl;
    int dim = 19 * 19; // 361
    Mat X_train1(training_set.size(), dim, CV_8UC1);
    
    for (int index = 0; index < (int)training_set.size(); index++) {
        Mat imgFlat = training_set.at(index).reshape(1, 1);
        imgFlat.copyTo(X_train1.row(index));
    }
    cout << "Training matrix: " << X_train1.rows << "x" << X_train1.cols << endl << endl;

    // Transpose and convert to float
    cout << "[4] Preparing data..." << endl;
    Mat XTrain_trans = X_train1.t();
    XTrain_trans.convertTo(XTrain_trans, CV_32FC1);
    
    // Compute mean
    Mat mean_train(dim, 1, CV_32FC1, Scalar(0));
    for (int i = 0; i < (int)training_set.size(); i++) {
        mean_train += XTrain_trans.col(i);
    }
    mean_train = mean_train / (float)training_set.size();
    
    // Visualize mean face
    Mat meanImg = mean_train.reshape(1, 19);
    Mat meanImg_vis;
    normalize(meanImg, meanImg_vis, 0, 255, NORM_MINMAX, CV_8UC1);
    resize(meanImg_vis, meanImg_vis, Size(200, 200));
    imwrite("images/train/eigenVecVis/mean_face.png", meanImg_vis);
    cout << "Saved mean face" << endl << endl;

    // Center data
    cout << "[5] Centering data..." << endl;
    for (int i = 0; i < (int)training_set.size(); i++) {
        XTrain_trans.col(i) = XTrain_trans.col(i) - mean_train;
    }
    cout << "Data centered" << endl << endl;

    // Compute covariance and SVD
    cout << "[6] Computing PCA..." << endl;
    Mat covariance_matrix;
    Mat meanMat;
    calcCovarMatrix(XTrain_trans, covariance_matrix, meanMat, COVAR_ROWS);
    covariance_matrix = covariance_matrix / training_set.size();
    
    Mat E, U, Vt;
    SVD::compute(covariance_matrix, E, U, Vt);
    Mat V = Vt.t();
    cout << "SVD computed" << endl;
    cout << "Eigenvalues: " << E.rows << endl;
    cout << "Eigenvectors: " << V.rows << "x" << V.cols << endl << endl;

    // Convert eigenvalues to vector
    vector<float> eigenvalues_vec;
    for (int i = 0; i < E.rows; i++) {
        if (E.depth() == CV_32F) {
            eigenvalues_vec.push_back(E.at<float>(i, 0));
        } else if (E.depth() == CV_64F) {
            eigenvalues_vec.push_back((float)E.at<double>(i, 0));
        }
    }
    
    // Find number of components for 90% variance
    float sumOfEigenValues = 0.0f;
    for (float val : eigenvalues_vec) {
        sumOfEigenValues += val;
    }
    
    vector<float> sumOfKEigenValues(dim, 0.f);
    for (int j = 0; j < dim; j++) {
        for (int k = 0; k <= j; k++) {
            sumOfKEigenValues[j] += eigenvalues_vec[k];
        }
    }

    int k_components = 0;
    for (int i = 0; i < (int)sumOfKEigenValues.size(); i++) {
        float variance_ratio = (float)sumOfKEigenValues[i] / sumOfEigenValues;
        if (variance_ratio < 0.9f) {
            k_components = i + 1;
        } else {
            break;
        }
    }
    
    cout << "[7] Visualizing eigenfaces..." << endl;
    cout << "Total eigenfaces available: " << V.rows << endl;
    cout << "Components for 90% variance: " << k_components << endl;
    cout << "Visualizing first 20 eigenfaces..." << endl << endl;

    // Create output directory
    system("mkdir -p images/train/eigenVecVis");

    // Visualize the top eigenfaces
    int num_to_visualize = min(20, k_components);
    for (int i = 0; i < num_to_visualize; i++) {
        Mat eigenvector = V.row(i).clone();
        
        // Convert to float if needed
        if (eigenvector.depth() != CV_32F) {
            eigenvector.convertTo(eigenvector, CV_32FC1);
        }
        
        // Reshape from 361-D to 19x19
        Mat eigenface = eigenvector.reshape(1, 19);
        
        // Normalize to 0-255 range
        Mat dst;
        normalize(eigenface, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        
        // Resize for better visualization
        resize(dst, dst, Size(200, 200));
        
        // Save as PNG
        string filename = "images/train/eigenVecVis/eigenface_" + to_string(i) + ".png";
        if (imwrite(filename, dst)) {
            cout << "✓ Eigenface " << i << " (variance ratio: " 
                 << (sumOfKEigenValues[i] / sumOfEigenValues) * 100 << "%)" << endl;
        }
    }
    
    cout << endl << "========================================" << endl;
    cout << "Visualization Complete!" << endl;
    cout << "========================================" << endl;
    cout << "Eigenfaces saved to: images/train/eigenVecVis/" << endl;
    cout << endl;
    
    // Create a grid visualization of the top eigenfaces
    cout << "[8] Creating grid visualization..." << endl;
    
    // Create a 4x5 grid of the first 20 eigenfaces
    Mat grid = Mat::zeros(4 * 200 + 30, 5 * 200 + 30, CV_8UC1);
    grid.setTo(255);  // White background
    
    int idx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 5; col++) {
            if (idx < num_to_visualize) {
                Mat eigenvector = V.row(idx).clone();
                if (eigenvector.depth() != CV_32F) {
                    eigenvector.convertTo(eigenvector, CV_32FC1);
                }
                
                Mat eigenface = eigenvector.reshape(1, 19);
                Mat dst;
                normalize(eigenface, dst, 0, 255, NORM_MINMAX, CV_8UC1);
                resize(dst, dst, Size(200, 200));
                
                int y = row * 200 + 10;
                int x = col * 200 + 10;
                dst.copyTo(grid(Rect(x, y, 200, 200)));
            }
            idx++;
        }
    }
    
    imwrite("images/train/eigenVecVis/eigenfaces_grid.png", grid);
    cout << "✓ Grid visualization saved to eigenfaces_grid.png" << endl;
    cout << endl;

    return 0;
}
