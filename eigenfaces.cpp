

#pragma GCC diagnostic ignored "-Wsystem-headers"
#pragma GCC diagnostic ignored "-Wno-parentheses"
#include <iostream>
#include <vector>
#include <istream>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Qt6 headers - QCustomPlot now installed

#include <QtCore/qglobal.h>
#include <QtCore/QVector>
#include <QMainWindow>
#include "/opt/homebrew/include/qcustomplot.h"

#define NO_OF_IMAGES 2429

using namespace std;
using namespace cv;

vector<Mat> read_faces() {
    vector<Mat> training_images;
    training_images.reserve(NO_OF_IMAGES);  // Pre-allocate
    string images_path = "images/train/face/face";
    string suffix = ".pgm";
    for (int i = 0; i < NO_OF_IMAGES; i++) {
        string filename = images_path + string(5 - to_string(i).length(), '0') + to_string(i) + suffix;
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Warning: Failed to load image: " << filename << endl;
            continue; // Skip missing image
        }
        training_images.push_back(img);
    }
    return training_images;
}

vector<Mat> extract_train_test_set(
        vector<Mat> faces,
        vector<Mat> &test_set) {

    int percentage_train = (0.9f * NO_OF_IMAGES);
    vector<Mat> training_set;
    training_set.reserve(percentage_train);
    test_set.reserve(NO_OF_IMAGES - percentage_train);

    for (int i = 0; i < percentage_train; i++) {
        training_set.push_back(faces[i]);
    }

    for (int i = percentage_train; i < NO_OF_IMAGES; i++) {
        test_set.push_back(faces[i]);
    }

    return training_set;
}

void minDistComputation(map<int, vector<double>>::iterator iter, vector<Point>& kNeighbours, int k, Point& minLoc, Point& maxLoc){
    int index = (*iter).first;
    vector<double> values = (*iter).second;
    double minValue;
    double maxValue;
    minMaxLoc(values, &minValue, &maxValue, &minLoc, &maxLoc);

    for(int i = 0; i < k; i++)
        kNeighbours.push_back(minLoc);
}

void calcNearestNeighbours(Mat& XTest, Mat& XTrain,
        vector<Mat>& nearestNeighbour, vector<int> testIndices,
        int training_examples_count, int kNeighbours, string space) {


    map<int, vector<double>> allDistances;
    vector<vector<double>> distancesVector(training_examples_count, vector<double>());
    
    for (int i = 0; i < (int)testIndices.size(); i++) {
        vector<double>& distances = distancesVector[i];
        distances.reserve(training_examples_count);
        
        for (int j = 0; j < training_examples_count; j++) {
            Scalar distance = norm(XTrain.col(j), XTest.col(testIndices[i]));
            distances.push_back(distance.val[0]);
        }
        allDistances[testIndices[i]] = distances;
    }

    vector<Point> kNNeighbours;
    Point minLoc;
    Point maxLoc;
    auto iter = allDistances.begin();

    while(iter != allDistances.end()) {
        minDistComputation(iter, kNNeighbours, kNeighbours, minLoc, maxLoc);
        cout << endl;
        cout << "Nearest training vector index to test vector for " + space + " is " << minLoc.x;
        iter++;
        cout << endl;
    }
}

void visualizeEigenVectors(Mat viz, const Mat& V,
        ofstream& file_eigen_vector_first_column, int index,
        vector<Mat>& eigenVectorViz) {

    viz = V.row(index);
    if (!viz.isContinuous()) {
        viz = viz.clone();
    }

    Mat rectangularMat = viz.reshape(1, 19);
    Mat dst;
    normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    resize(dst, dst, Size(200, 200));

    eigenVectorViz.push_back(dst);
    imwrite("images/train/eigenVecVis/eigenVecViz" + to_string(index) + ".png", dst);
}

// Qt plotting disabled - QCustomPlot not installed

void plotDistances(QCustomPlot &customPlot, const QVector<double>& x, const QVector<double>& y, QMainWindow &window, int plotNumber) {
    customPlot.addGraph();
    customPlot.graph(0)->setData(x, y);
    customPlot.xAxis->setLabel("Index");
    customPlot.yAxis->setLabel("Distance");
    customPlot.rescaleAxes();
    customPlot.setWindowTitle(QString("Distance Plot %1").arg(plotNumber));
    window.setCentralWidget(&customPlot);
    window.resize(600, 400);
    window.show();
}

int main(int argc, char **argv) {
    // Ensure program is run from project root
    std::ifstream test_image_file("images/train/face/face00000.pgm");
    if (!test_image_file.good()) {
        std::cerr << "\nERROR: Please run this program from the project root directory (where images/ is located)." << std::endl;
        std::cerr << "Current working directory does not contain images/train/face/face00000.pgm" << std::endl;
        return 1;
    }
    test_image_file.close();


    QApplication a(argc, argv);
    QMainWindow window1[10];
    QCustomPlot customPlot1[10];
    for (int i = 0; i < 10; i++)
        window1[i].setCentralWidget(&customPlot1[i]);

    QCustomPlot customPlot2[10];
    QMainWindow window2[10];
    for (int i = 0; i < 10; i++)
        window2[i].setCentralWidget(&customPlot2[i]);

    vector<Mat> eigenVectorViz;

    // Reading faces into a vector of matrices
    vector<Mat> faces = read_faces();
    // Use std::shuffle instead of deprecated random_shuffle
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(faces.begin(), faces.end(), std::default_random_engine(seed));
    cout << faces.size() << endl;

    vector<Mat> training_set;
    vector<Mat> test_set;
    training_set = extract_train_test_set(faces, test_set);

    cout << " Training set size " << training_set.size() << endl;
    cout << " Test set size " << test_set.size() << endl;

    int dim = training_set[0].rows * training_set[0].cols;

    // OPTIMIZATION 1: Use reshape instead of triple nested loop
    Mat X_train1(2186, 361, CV_8UC1);
    for (int index = 0; index < (int)training_set.size(); index++) {
        Mat imgFlat = training_set.at(index).reshape(1, 1);
        imgFlat.copyTo(X_train1.row(index));
    }

    cout << "Height " << X_train1.rows << " Width " << X_train1.cols << endl;

    Mat XTrain_trans = X_train1.t();
    cout << "Rows " << XTrain_trans.rows << " Cols " << XTrain_trans.cols << endl;

    Mat sum_train = Mat::zeros(dim, 1, CV_32FC1);
    Mat example_col(dim, 1, CV_32FC1);
    Mat mean_train;

    for (int index = 0; index < (int)training_set.size(); index++) {
        XTrain_trans.col(index).convertTo(example_col, CV_32FC1);
        sum_train += example_col;
    }

    divide(sum_train, training_set.size(), mean_train);

    Mat samples = XTrain_trans.clone();
    XTrain_trans.convertTo(XTrain_trans, CV_32FC1);

    // Centering the training matrix
    for (int index = 0; index < (int)training_set.size(); index++) {
        XTrain_trans.col(index) = XTrain_trans.col(index) - mean_train;
    }

    string file_path_center = "images/train/X_center.dat";
    ofstream file_handle_center(file_path_center.c_str(), ios::trunc);
    file_handle_center << XTrain_trans;
    file_handle_center.close();  // Close immediately after use
    cout << "Dimensions - Rows " << XTrain_trans.rows << " Columns " << XTrain_trans.cols << endl;

    Mat distance_matrix = Mat::zeros(dim, dim, CV_32FC1);
    Mat covariance_matrix = Mat::zeros(dim, dim, CV_32FC1);

    distance_matrix = XTrain_trans * XTrain_trans.t();
    covariance_matrix = distance_matrix / training_set.size();

    string file_path_covariance = "images/train/X_covariance.dat";
    ofstream file_handle_covariance(file_path_covariance.c_str(), ios::trunc);
    file_handle_covariance << covariance_matrix;
    file_handle_covariance.close();  // Close immediately after use

    cout << "**CO-VARIANCE COMPUTED**" << endl;
    cout << "Co-variance dimensions rows " << covariance_matrix.rows << " columns " << covariance_matrix.cols << endl;

    Mat E, V;
    eigen(covariance_matrix, E, V);

    // OPTIMIZATION 2: Fix meanImg calculation with proper reset
    Mat meanImg(19, 19, CV_32FC1);
    int matIndex = 0;  // Reset at start
    for (int i = 0; i < meanImg.rows; i++) {
        for (int j = 0; j < meanImg.cols; j++) {
            meanImg.at<float>(i, j) = floorf(mean_train.at<float>(matIndex));
            matIndex++;
        }
    }
    matIndex = 0;  // Reset after use

    meanImg = meanImg / 255;

    string file_path_eigen_values = "images/train/X_eigen_values.dat";
    ofstream file_handle_eigen_values(file_path_eigen_values.c_str(), ios::trunc);
    file_handle_eigen_values << E;
    file_handle_eigen_values.close();  // Close immediately

    // OPTIMIZATION 3: Pre-reserve vector space
    std::vector<double> eigenValues;
    eigenValues.reserve(E.rows);
    std::vector<double> basis;
    basis.reserve(E.rows);

    for(int i = 0; i < E.rows; i++){
        double value = E.at<float>(i);
        eigenValues.push_back(value);
        basis.push_back(i);
    }

    // Qt spectrum plot disabled - QCustomPlot not installed
    // QCustomPlot spectrumPlot;
    // QMainWindow window;
    // ... Qt plotting code removed

    Mat coVarMat;
    Mat meanMat;

    string file_path_covar = "images/coVarMat.dat";
    ofstream file_handle_coVar(file_path_covar.c_str(), ios::trunc);

    calcCovarMatrix(XTrain_trans, coVarMat, meanMat, COVAR_ROWS);
    coVarMat = coVarMat / training_set.size();
    file_handle_coVar << coVarMat << endl << endl << endl;
    file_handle_coVar << "*******************MEAN matrix**************************\n";
    file_handle_coVar << "Columns " << meanMat.cols << "\t" << "Rows " << meanMat.rows << endl << endl << endl;
    file_handle_coVar << meanMat << endl;
    file_handle_coVar.close();

    resize(meanImg, meanImg, Size(200, 200));

    cout << "Eigen vector dimensions rows " << V.rows << " columns " << V.cols << endl;

    float sumOfEigenValues = 0.0f;
    for (int i = 0; i < E.rows; i++) {
        sumOfEigenValues += E.at<float>(i);
    }
    cout << "Sum of all " << dim << " eigen values is " << sumOfEigenValues << endl;

    vector<float> sumOfKEigenValues(dim, 0.f);
    for (int j = 0; j < dim; j++) {
        for (int k = 0; k <= j; k++) {
            sumOfKEigenValues[j] += E.at<float>(k);
        }
    }

    cout << "Size of sum of k eigen values array " << sumOfKEigenValues.size() << endl;

    vector<float> smallestKEigenValues;
    smallestKEigenValues.reserve(dim);
    auto iterK = sumOfKEigenValues.begin();
    
    while (iterK != sumOfKEigenValues.end()
            && (float) (*(iterK) / sumOfEigenValues) < 0.9 ) {
        cout << (float) (*iterK) / sumOfEigenValues;
        smallestKEigenValues.push_back((float) (*iterK) / sumOfEigenValues);
        iterK++;
        cout << endl;
    }

    if (smallestKEigenValues.empty()) {
        cerr << "Error: No eigenvalues satisfy the 0.9 threshold criterion." << endl;
        return 1;
    }

    Mat viz(19, 19, CV_32FC1);
    string file_eigen_vector_first_column_path = "images/train/X_eigen_vector_first_col.dat";
    ofstream file_eigen_vector_first_column(file_eigen_vector_first_column_path.c_str(), ios::trunc);

    int ind = 0;
    while (ind < (int)smallestKEigenValues.size()) {
        visualizeEigenVectors(viz, V, file_eigen_vector_first_column, ind, eigenVectorViz);
        ind++;
    }

    // OPTIMIZATION 1: Test set with reshape
    string file_test_file_path = "images/test/X_test.dat";
    ofstream file_handle_test_file_stream(file_test_file_path, ios::trunc);

    Mat X_test1(243, 361, CV_8UC1);
    for (int index = 0; index < (int)test_set.size(); index++) {
        Mat imgFlat = test_set.at(index).reshape(1, 1);
        imgFlat.copyTo(X_test1.row(index));
        
        for (int k = 0; k < dim; k++) {
            file_handle_test_file_stream << (int)imgFlat.at<uchar>(0, k);
            if (k < dim - 1) file_handle_test_file_stream << ",";
        }
        file_handle_test_file_stream << endl;
    }
    file_handle_test_file_stream.close();

    Mat XTest_trans = X_test1.t();
    XTest_trans.convertTo(XTest_trans, CV_32FC1);

    for (int index = 0; index < (int)test_set.size(); index++) {
        XTest_trans.col(index) = XTest_trans.col(index) - mean_train;
    }

    RNG rng;
    int randIndex;
    Mat X_rand_test_samples(10, 361, CV_8UC1);
    vector<int> testSamplesIndices;
    testSamplesIndices.reserve(10);

    for (int i = 0; i < 10; i++) {
        randIndex = rng.uniform(0, 243);
        testSamplesIndices.push_back(randIndex);
        X_rand_test_samples.row(i) = X_test1.row(randIndex);
    }

    string file_X_rand_test_samples = "images/test/X_rand_test_samples_trans";
    ofstream file_X_rand_test_samples_handle(file_X_rand_test_samples.c_str(), ios::trunc);

    Mat X_rand_test_samples_trans = X_rand_test_samples;
    X_rand_test_samples_trans = X_rand_test_samples_trans.t();
    X_rand_test_samples_trans.convertTo(X_rand_test_samples_trans, CV_32FC1);

    string file_test_train_file_path = "images/test/X_test_train_distance.dat";
    ofstream file_handle_test_train_file_stream(file_test_train_file_path, ios::trunc);
    
    vector<float> EucDistToTrain_Data;
    EucDistToTrain_Data.reserve(10 * 2186);

    vector<double> column1;
    vector<double> column2;

    // OPTIMIZATION 4: Cache matrix columns
    for (int i = 0; i < 10; i++) {
        file_X_rand_test_samples_handle << X_rand_test_samples_trans.col(i) << endl;
        column1.reserve(2186);
        column2.reserve(2186);

        Mat testCol = XTest_trans.col(testSamplesIndices[i]);
        
        for (int j = 0; j < 2186; j++) {
            Mat trainCol = XTrain_trans.col(j);
            Scalar value = norm(testCol, trainCol, NORM_L2);
            EucDistToTrain_Data.push_back(value.val[0]);
            column1.push_back(j);
            column2.push_back(value.val[0]);
        }
        
        sort(column2.begin(), column2.end(), greater<float>());
        QVector<double> qx(column1.begin(), column1.end());
        QVector<double> qy(column2.begin(), column2.end());
        plotDistances(customPlot1[i], qx, qy, window1[i], i);
        column1.clear();
        column2.clear();
    }

    sort(EucDistToTrain_Data.begin(), EucDistToTrain_Data.end(), greater<float>());

    for (auto& val : EucDistToTrain_Data) {
        file_handle_test_train_file_stream << val << endl;
    }

    // Projecting all training samples into the PCA subspace
    Mat projectionMatrix_trainingData(20, 2186, CV_32FC1);
    Mat smallestKEigenVectors = V.rowRange(0, 20);

    projectionMatrix_trainingData = smallestKEigenVectors * XTrain_trans;

    string file_projection_matrix = "images/projection_matrix_trainingData.dat";
    ofstream projection_matrix_file_handle(file_projection_matrix, ios::trunc);
    cout << "Projection matrix for training data rows " << projectionMatrix_trainingData.rows 
         << " and columns " << projectionMatrix_trainingData.cols << endl;
    projection_matrix_file_handle << projectionMatrix_trainingData << endl;
    projection_matrix_file_handle.close();

    string file_projection_matrix_testData = "images/projection_matrix_testData.dat";
    ofstream projection_matrix_file_handle_testData(file_projection_matrix_testData, ios::trunc);

    projection_matrix_file_handle_testData << "****Projection of test data onto subspace spanned by k eigen vectors****" << endl << endl;

    Mat projectionMatrix_testData(20, 243, CV_32FC1);
    projectionMatrix_testData = smallestKEigenVectors * XTest_trans;
    projection_matrix_file_handle_testData << projectionMatrix_testData << endl << endl << endl << endl << endl;
    projection_matrix_file_handle_testData.close();

    cout << "Projection matrix for test data rows " << projectionMatrix_testData.rows 
         << " and columns " << projectionMatrix_testData.cols << endl;

    vector<double> column_test_set_indices;
    vector<double> column_distances;
    vector<float> EucDistToTrain_Data_after_pca;
    EucDistToTrain_Data_after_pca.reserve(10 * 2186);

    for (int i = 0; i < (int)testSamplesIndices.size(); i++) {
        column_test_set_indices.reserve(2186);
        column_distances.reserve(2186);

        Mat testCol = projectionMatrix_testData.col(testSamplesIndices[i]);
        
        for (int j = 0; j < (int)training_set.size(); j++) {
            Mat trainCol = projectionMatrix_trainingData.col(j);
            Scalar value = norm(testCol, trainCol, NORM_L2);
            EucDistToTrain_Data_after_pca.push_back(value.val[0]);
            column_test_set_indices.push_back(j);
            column_distances.push_back(value.val[0]);
        }
        
        sort(column_distances.begin(), column_distances.end(), greater<float>());
        QVector<double> qx(column_test_set_indices.begin(), column_test_set_indices.end());
        QVector<double> qy(column_distances.begin(), column_distances.end());
        plotDistances(customPlot2[i], qx, qy, window2[i], i);
        column_distances.clear();
        column_test_set_indices.clear();
    }

    sort(EucDistToTrain_Data_after_pca.begin(), EucDistToTrain_Data_after_pca.end(), greater<float>());

    string file_test_train_file_after_pca_path = "images/test/X_test_train_distance_after_pca.dat";
    ofstream file_handle_test_train_file_after_pca_stream(file_test_train_file_after_pca_path, ios::trunc);
    
    for (auto& val : EucDistToTrain_Data_after_pca) {
        file_handle_test_train_file_after_pca_stream << val << endl;
    }

    cout << "Size of euclidean distance to train data after pca " << EucDistToTrain_Data_after_pca.size() << endl;

    // Nearest neighbours
    vector<Mat> nearestNeighbours_orig_space;
    calcNearestNeighbours(XTest_trans, XTrain_trans, nearestNeighbours_orig_space, testSamplesIndices, training_set.size(), 1, "Original Space");
    calcNearestNeighbours(projectionMatrix_testData, projectionMatrix_trainingData, nearestNeighbours_orig_space, testSamplesIndices, training_set.size(), 1, "Sub-space");

    imshow("Mean Img", meanImg);
    // Start Qt event loop so Qt windows appear immediately
    int qt_result = a.exec();

    // After Qt windows are closed, optionally wait for OpenCV window
    waitKey(0);

    // Close all file handles
    file_eigen_vector_first_column.close();
    file_X_rand_test_samples_handle.close();
    file_handle_test_train_file_stream.close();
    file_handle_test_train_file_after_pca_stream.close();

    // Release matrices
    X_train1.release();
    XTrain_trans.release();
    covariance_matrix.release();
    viz.release();
    E.release();
    V.release();

    return qt_result;
}