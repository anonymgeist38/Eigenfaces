#include <iostream>
#include <vector>
#include <istream>
#include <fstream>
#include <random>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define NO_OF_IMAGES 2429

using namespace std;
using namespace cv;

vector<Mat> read_faces() {
    vector<Mat> training_images;
    training_images.reserve(NO_OF_IMAGES);
    string images_path = "images/train/face";
    string suffix = ".pgm";
    for (int i = 0; i < NO_OF_IMAGES; i++) {
        Mat img = imread(cv::format("%s/face%05d.pgm", images_path.c_str(), i), 0);
        if (img.empty()) {
            cerr << "Failed to load image: " << cv::format("%s/face%05d.pgm", images_path.c_str(), i) << endl;
            continue;
        }
        training_images.push_back(img);
    }
    cout << "Loaded " << training_images.size() << " images" << endl;
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

int main() {
    cout << "========================================" << endl;
    cout << "Eigenfaces - Face Recognition System" << endl;
    cout << "========================================" << endl << endl;

    // Step 1: Read faces
    cout << "[1] Loading face images..." << endl;
    vector<Mat> faces = read_faces();
    
    if (faces.empty()) {
        cerr << "Error: No faces loaded!" << endl;
        return 1;
    }

    cout << "Total images loaded: " << faces.size() << endl;
    cout << "Image dimensions: " << faces[0].rows << "x" << faces[0].cols << endl << endl;

    // Step 2: Split into train/test
    cout << "[2] Splitting data into train/test sets (90/10)..." << endl;
    vector<Mat> test_set;
    vector<Mat> training_set = extract_train_test_set(faces, test_set);
    
    cout << "Training set size: " << training_set.size() << endl;
    cout << "Test set size: " << test_set.size() << endl << endl;

    // Step 3: Construct training matrix
    cout << "[3] Constructing training matrix..." << endl;
    int dim = 19 * 19; // 361
    Mat X_train1(training_set.size(), dim, CV_8UC1);
    
    for (int index = 0; index < (int)training_set.size(); index++) {
        Mat imgFlat = training_set.at(index).reshape(1, 1);
        imgFlat.copyTo(X_train1.row(index));
    }
    
    cout << "Training matrix shape: " << X_train1.rows << "x" << X_train1.cols << endl;
    cout << "Data type: " << X_train1.type() << endl << endl;

    // Step 4: Transpose and convert to float
    cout << "[4] Transposing and converting to float..." << endl;
    Mat XTrain_trans = X_train1.t();
    XTrain_trans.convertTo(XTrain_trans, CV_32FC1);
    cout << "Transposed matrix shape: " << XTrain_trans.rows << "x" << XTrain_trans.cols << endl << endl;

    // Step 5: Compute mean
    cout << "[5] Computing mean face..." << endl;
    Mat mean_train(dim, 1, CV_32FC1, Scalar(0));
    for (int i = 0; i < (int)training_set.size(); i++) {
        mean_train += XTrain_trans.col(i);
    }
    mean_train = mean_train / (float)training_set.size();
    cout << "Mean computed" << endl << endl;

    // Step 6: Center data
    cout << "[6] Centering data..." << endl;
    for (int i = 0; i < (int)training_set.size(); i++) {
        XTrain_trans.col(i) = XTrain_trans.col(i) - mean_train;
    }
    cout << "Data centered" << endl << endl;

    // Step 7: Compute covariance matrix and SVD
    cout << "[7] Computing covariance matrix..." << endl;
    Mat covariance_matrix;
    Mat meanMat;
    calcCovarMatrix(XTrain_trans, covariance_matrix, meanMat, COVAR_ROWS);
    covariance_matrix = covariance_matrix / training_set.size();
    cout << "Covariance matrix shape: " << covariance_matrix.rows << "x" << covariance_matrix.cols << endl;
    cout << "Memory usage: " << (covariance_matrix.total() * covariance_matrix.elemSize() / 1024.0 / 1024.0) << " MB" << endl << endl;

    cout << "[8] Computing SVD decomposition..." << endl;
    Mat E, U, Vt;
    SVD::compute(covariance_matrix, E, U, Vt);
    Mat V = Vt.t();  // Transpose to get eigenvectors as columns
    cout << "Eigenvalues shape: " << E.rows << "x" << E.cols << endl;
    cout << "Eigenvectors shape: " << V.rows << "x" << V.cols << endl << endl;

    // Step 9: Compute explained variance
    cout << "[9] Analyzing eigenvalue spectrum..." << endl;
    
    // Convert eigenvalues to vector for easier manipulation
    // E is a column vector, access values safely
    vector<float> eigenvalues_vec;
    for (int i = 0; i < E.rows; i++) {
        if (E.depth() == CV_32F) {
            eigenvalues_vec.push_back(E.at<float>(i, 0));
        } else if (E.depth() == CV_64F) {
            eigenvalues_vec.push_back((float)E.at<double>(i, 0));
        }
    }
    
    float sumOfEigenValues = 0.0f;
    for (float val : eigenvalues_vec) {
        sumOfEigenValues += val;
    }
    cout << "Sum of eigenvalues: " << sumOfEigenValues << endl;

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
    
    cout << "Number of components for 90% variance: " << k_components << endl << endl;

    // Step 10: Project data
    cout << "[10] Projecting training data..." << endl;
    Mat projectionMatrix = V.rowRange(0, k_components).clone();
    projectionMatrix.convertTo(projectionMatrix, CV_32FC1);  // Ensure float type
    Mat projectionMatrix_trainingData = projectionMatrix * XTrain_trans;
    cout << "Projected training data shape: " << projectionMatrix_trainingData.rows << "x" << projectionMatrix_trainingData.cols << endl << endl;

    // Step 11: Process test set
    cout << "[11] Processing test set..." << endl;
    Mat X_test1(test_set.size(), dim, CV_8UC1);
    for (int index = 0; index < (int)test_set.size(); index++) {
        Mat imgFlat = test_set.at(index).reshape(1, 1);
        imgFlat.copyTo(X_test1.row(index));
    }
    
    Mat XTest_trans = X_test1.t();
    XTest_trans.convertTo(XTest_trans, CV_32FC1);
    
    for (int index = 0; index < (int)test_set.size(); index++) {
        XTest_trans.col(index) = XTest_trans.col(index) - mean_train;
    }
    
    Mat projectionMatrix_testData = projectionMatrix * XTest_trans;
    cout << "Projected test data shape: " << projectionMatrix_testData.rows << "x" << projectionMatrix_testData.cols << endl << endl;

    // Step 12: k-NN classification
    cout << "[12] Computing k-NN classification (k=1)..." << endl;
    vector<float> distances;
    distances.reserve(test_set.size() * training_set.size());
    
    for (int i = 0; i < (int)test_set.size(); i++) {
        Mat testCol = projectionMatrix_testData.col(i);
        float min_dist = FLT_MAX;
        int nn_index = 0;
        
        for (int j = 0; j < (int)training_set.size(); j++) {
            Mat trainCol = projectionMatrix_trainingData.col(j);
            float dist = (float)norm(testCol, trainCol, NORM_L2);
            distances.push_back(dist);
            
            if (dist < min_dist) {
                min_dist = dist;
                nn_index = j;
            }
        }
        
        if (i < 5) {
            cout << "  Test sample " << i << ": nearest neighbor is training sample " << nn_index 
                 << " (distance: " << min_dist << ")" << endl;
        }
    }
    cout << "Classification complete" << endl << endl;

    // Step 13: Save results
    cout << "[13] Saving results to file..." << endl;
    string output_file = "images/test/classification_results.dat";
    ofstream results_file(output_file, ios::trunc);
    
    for (auto& dist : distances) {
        results_file << dist << endl;
    }
    results_file.close();
    
    cout << "Results saved to: " << output_file << endl;
    cout << "Total distances computed: " << distances.size() << endl << endl;

    // Summary
    cout << "========================================" << endl;
    cout << "Eigenfaces Analysis Complete!" << endl;
    cout << "========================================" << endl;
    cout << "Summary:" << endl;
    cout << "  - Loaded " << training_set.size() << " training images" << endl;
    cout << "  - Tested on " << test_set.size() << " test images" << endl;
    cout << "  - Used " << k_components << " PCA components (90% variance)" << endl;
    cout << "  - Computed " << distances.size() << " distances" << endl;
    cout << endl;

    return 0;
}
