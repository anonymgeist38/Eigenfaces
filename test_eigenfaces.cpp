#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

// ============================================================================
// Test Fixtures and Helper Functions
// ============================================================================

class EigenfacesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create synthetic face images for testing
        img_19x19_1 = Mat::zeros(19, 19, CV_8UC1);
        img_19x19_2 = Mat::ones(19, 19, CV_8UC1) * 100;
        img_19x19_3 = Mat::ones(19, 19, CV_8UC1) * 200;
    }

    Mat img_19x19_1;
    Mat img_19x19_2;
    Mat img_19x19_3;
};

// ============================================================================
// Test Cases for Image Dimensions and Properties
// ============================================================================

TEST_F(EigenfacesTest, ImageDimensionsAre19x19) {
    EXPECT_EQ(img_19x19_1.rows, 19);
    EXPECT_EQ(img_19x19_1.cols, 19);
    EXPECT_EQ(img_19x19_2.rows, 19);
    EXPECT_EQ(img_19x19_2.cols, 19);
}

TEST_F(EigenfacesTest, ImageVectorDimensionIs361) {
    int dim = img_19x19_1.rows * img_19x19_1.cols;
    EXPECT_EQ(dim, 361);
}

TEST_F(EigenfacesTest, ImageTypeIsSingleChannelUnsignedChar) {
    EXPECT_EQ(img_19x19_1.type(), CV_8UC1);
    EXPECT_EQ(img_19x19_2.type(), CV_8UC1);
}

// ============================================================================
// Test Cases for Training/Test Set Split (90/10)
// ============================================================================

TEST_F(EigenfacesTest, TrainingTestSplitProportions) {
    const int NO_OF_IMAGES = 2429;
    int training_size = static_cast<int>(0.9f * NO_OF_IMAGES);
    int test_size = NO_OF_IMAGES - training_size;
    
    EXPECT_EQ(training_size, 2186);
    EXPECT_EQ(test_size, 243);
    EXPECT_EQ(training_size + test_size, NO_OF_IMAGES);
}

TEST_F(EigenfacesTest, TrainingSetSizeLargerThanTestSet) {
    const int NO_OF_IMAGES = 2429;
    int training_size = static_cast<int>(0.9f * NO_OF_IMAGES);
    int test_size = NO_OF_IMAGES - training_size;
    
    EXPECT_GT(training_size, test_size);
}

// ============================================================================
// Test Cases for Matrix Construction from Images
// ============================================================================

TEST_F(EigenfacesTest, ConstructTrainingMatrixDimensions) {
    vector<Mat> training_images;
    training_images.push_back(img_19x19_1);
    training_images.push_back(img_19x19_2);
    training_images.push_back(img_19x19_3);
    
    int num_images = training_images.size();
    int dim = 19 * 19;
    
    Mat X_train(num_images, dim, CV_8UC1);
    
    EXPECT_EQ(X_train.rows, 3);
    EXPECT_EQ(X_train.cols, 361);
}

TEST_F(EigenfacesTest, TransposedMatrixDimensions) {
    vector<Mat> training_images;
    training_images.push_back(img_19x19_1);
    training_images.push_back(img_19x19_2);
    
    Mat X_train(2, 361, CV_8UC1);
    Mat XTrain_trans = X_train.t();
    
    EXPECT_EQ(XTrain_trans.rows, 361);
    EXPECT_EQ(XTrain_trans.cols, 2);
}

// ============================================================================
// Test Cases for Mean Calculation
// ============================================================================

TEST_F(EigenfacesTest, MeanOfZeroMatrixIsZero) {
    Mat test_matrix = Mat::zeros(5, 10, CV_32FC1);
    Mat mean;
    
    // Calculate mean
    Mat sum = Mat::zeros(10, 1, CV_32FC1);
    for (int i = 0; i < 5; i++) {
        Mat col = test_matrix.row(i).t();
        col.convertTo(col, CV_32FC1);
        sum += col;
    }
    divide(sum, 5, mean);
    
    // All elements should be zero
    double min_val, max_val;
    minMaxLoc(mean, &min_val, &max_val);
    EXPECT_DOUBLE_EQ(min_val, 0.0);
    EXPECT_DOUBLE_EQ(max_val, 0.0);
}

TEST_F(EigenfacesTest, MeanOfUniformMatrixIsCorrect) {
    Mat test_matrix = Mat::ones(5, 10, CV_32FC1) * 100;
    Mat mean;
    
    Mat sum = Mat::zeros(10, 1, CV_32FC1);
    for (int i = 0; i < 5; i++) {
        sum += test_matrix.row(i).t();
    }
    divide(sum, 5, mean);
    
    // All elements should be 100
    double min_val, max_val;
    minMaxLoc(mean, &min_val, &max_val);
    EXPECT_DOUBLE_EQ(min_val, 100.0);
    EXPECT_DOUBLE_EQ(max_val, 100.0);
}

// ============================================================================
// Test Cases for Centering/Normalization
// ============================================================================

TEST_F(EigenfacesTest, CenteredMatrixHasMeanZero) {
    Mat original = Mat::ones(5, 10, CV_32FC1) * 50;
    Mat centered = original.clone();
    
    // Calculate mean
    Mat sum = Mat::zeros(10, 1, CV_32FC1);
    for (int i = 0; i < 5; i++) {
        sum += original.row(i).t();
    }
    Mat mean;
    divide(sum, 5, mean);
    
    // Center the matrix
    for (int i = 0; i < 5; i++) {
        centered.row(i) = centered.row(i) - mean.t();
    }
    
    // Verify mean is approximately zero
    Mat centered_sum = Mat::zeros(10, 1, CV_32FC1);
    for (int i = 0; i < 5; i++) {
        centered_sum += centered.row(i).t();
    }
    Mat centered_mean;
    divide(centered_sum, 5, centered_mean);
    
    double max_val;
    minMaxLoc(abs(centered_mean), nullptr, &max_val);
    EXPECT_LT(max_val, 1e-5);
}

// ============================================================================
// Test Cases for Distance Computation (k-NN)
// ============================================================================

TEST_F(EigenfacesTest, EuclideanDistanceBetweenIdenticalVectors) {
    Mat vec1 = Mat::ones(361, 1, CV_32FC1) * 5.0;
    Mat vec2 = Mat::ones(361, 1, CV_32FC1) * 5.0;
    
    double distance = norm(vec1 - vec2);
    EXPECT_DOUBLE_EQ(distance, 0.0);
}

TEST_F(EigenfacesTest, EuclideanDistanceBetweenOrthogonalVectors) {
    Mat vec1 = Mat::zeros(4, 1, CV_32FC1);
    Mat vec2 = Mat::zeros(4, 1, CV_32FC1);
    
    vec1.at<float>(0, 0) = 3.0;
    vec1.at<float>(1, 0) = 4.0;
    
    vec2.at<float>(2, 0) = 3.0;
    vec2.at<float>(3, 0) = 4.0;
    
    double distance = norm(vec1 - vec2);
    EXPECT_DOUBLE_EQ(distance, sqrt(50.0));
}

TEST_F(EigenfacesTest, DistanceIsSymmetric) {
    Mat vec1 = Mat::ones(10, 1, CV_32FC1) * 2.0;
    Mat vec2 = Mat::ones(10, 1, CV_32FC1) * 5.0;
    
    double dist_1_2 = norm(vec1 - vec2);
    double dist_2_1 = norm(vec2 - vec1);
    
    EXPECT_DOUBLE_EQ(dist_1_2, dist_2_1);
}

// ============================================================================
// Test Cases for Data Integrity
// ============================================================================

TEST_F(EigenfacesTest, ImageVectorNotEmpty) {
    vector<Mat> images;
    images.push_back(img_19x19_1);
    
    EXPECT_FALSE(images.empty());
    EXPECT_EQ(images.size(), 1);
}

TEST_F(EigenfacesTest, VectorElementAccess) {
    vector<Mat> images;
    images.push_back(img_19x19_1);
    images.push_back(img_19x19_2);
    
    Mat first = images.at(0);
    Mat second = images.at(1);
    
    EXPECT_EQ(first.rows, 19);
    EXPECT_EQ(second.rows, 19);
}

TEST_F(EigenfacesTest, MultipleImagesStored) {
    vector<Mat> images;
    images.push_back(img_19x19_1);
    images.push_back(img_19x19_2);
    images.push_back(img_19x19_3);
    
    EXPECT_EQ(images.size(), 3);
}

// ============================================================================
// Test Cases for Matrix Operations
// ============================================================================

TEST_F(EigenfacesTest, MatrixTransposeInvertsRowsCols) {
    Mat original(5, 361, CV_32FC1);
    Mat transposed = original.t();
    
    EXPECT_EQ(transposed.rows, 361);
    EXPECT_EQ(transposed.cols, 5);
}

TEST_F(EigenfacesTest, DataTypeConversionUchar2Float) {
    Mat uchar_mat = Mat::ones(19, 19, CV_8UC1) * 200;
    Mat float_mat;
    
    uchar_mat.convertTo(float_mat, CV_32FC1);
    
    EXPECT_EQ(float_mat.type(), CV_32FC1);
    EXPECT_EQ(float_mat.at<float>(0, 0), 200.0f);
}

TEST_F(EigenfacesTest, ZeroMatrixCreation) {
    Mat zero_mat = Mat::zeros(19, 19, CV_32FC1);
    
    double min_val, max_val;
    minMaxLoc(zero_mat, &min_val, &max_val);
    
    EXPECT_DOUBLE_EQ(min_val, 0.0);
    EXPECT_DOUBLE_EQ(max_val, 0.0);
}

TEST_F(EigenfacesTest, OnesMatrixCreation) {
    Mat ones_mat = Mat::ones(10, 10, CV_32FC1);
    
    double min_val, max_val;
    minMaxLoc(ones_mat, &min_val, &max_val);
    
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 1.0);
}

// ============================================================================
// Test Cases for Covariance/Correlation Matrix Properties
// ============================================================================

TEST_F(EigenfacesTest, CovarianceMatrixIsSquare) {
    int dim = 361;
    Mat cov_matrix = Mat::zeros(dim, dim, CV_32FC1);
    
    EXPECT_EQ(cov_matrix.rows, cov_matrix.cols);
}

TEST_F(EigenfacesTest, CovarianceMatrixSymmetry) {
    Mat test_mat = Mat(10, 5, CV_32FC1);
    randu(test_mat, 0, 255);
    
    Mat cov;
    Mat mean_val;
    calcCovarMatrix(test_mat, cov, mean_val, COVAR_ROWS | COVAR_NORMAL);
    
    // Check if cov == cov.t() (transpose)
    Mat diff = cov - cov.t();
    double max_val;
    minMaxLoc(abs(diff), nullptr, &max_val);
    
    EXPECT_LT(max_val, 1e-5);
}

// ============================================================================
// Test Cases for Image Conversion
// ============================================================================

TEST_F(EigenfacesTest, ImagePixelValueRange) {
    EXPECT_GE(img_19x19_1.at<uchar>(0, 0), 0);
    EXPECT_LE(img_19x19_2.at<uchar>(0, 0), 255);
}

TEST_F(EigenfacesTest, AllImagesHaveSameDimensions) {
    vector<Mat> images;
    images.push_back(img_19x19_1);
    images.push_back(img_19x19_2);
    images.push_back(img_19x19_3);
    
    for (const auto& img : images) {
        EXPECT_EQ(img.rows, 19);
        EXPECT_EQ(img.cols, 19);
    }
}

// ============================================================================
// Test Cases for Edge Cases
// ============================================================================

TEST_F(EigenfacesTest, SingleImageProcessing) {
    vector<Mat> single_image;
    single_image.push_back(img_19x19_1);
    
    EXPECT_EQ(single_image.size(), 1);
    EXPECT_FALSE(single_image.empty());
}

TEST_F(EigenfacesTest, LargeNumberOfImages) {
    vector<Mat> images;
    for (int i = 0; i < 2429; i++) {
        images.push_back(Mat::zeros(19, 19, CV_8UC1));
    }
    
    EXPECT_EQ(images.size(), 2429);
}

TEST_F(EigenfacesTest, DataConsistencyAfterTranspose) {
    Mat original = Mat::ones(2, 3, CV_32FC1);
    original.at<float>(0, 0) = 1.0;
    original.at<float>(0, 1) = 2.0;
    original.at<float>(0, 2) = 3.0;
    original.at<float>(1, 0) = 4.0;
    original.at<float>(1, 1) = 5.0;
    original.at<float>(1, 2) = 6.0;
    
    Mat transposed = original.t();
    
    EXPECT_EQ(transposed.at<float>(0, 0), 1.0);
    EXPECT_EQ(transposed.at<float>(1, 0), 2.0);
    EXPECT_EQ(transposed.at<float>(2, 0), 3.0);
    EXPECT_EQ(transposed.at<float>(0, 1), 4.0);
    EXPECT_EQ(transposed.at<float>(1, 1), 5.0);
    EXPECT_EQ(transposed.at<float>(2, 1), 6.0);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(EigenfacesTest, EndToEndMatrixConstruction) {
    vector<Mat> training_images;
    training_images.push_back(img_19x19_1);
    training_images.push_back(img_19x19_2);
    
    int num_images = training_images.size();
    int dim = 361;
    
    Mat X_train(num_images, dim, CV_8UC1);
    
    // Verify dimensions
    EXPECT_EQ(X_train.rows, 2);
    EXPECT_EQ(X_train.cols, 361);
    
    // Transpose
    Mat XTrain_trans = X_train.t();
    
    EXPECT_EQ(XTrain_trans.rows, 361);
    EXPECT_EQ(XTrain_trans.cols, 2);
}

TEST_F(EigenfacesTest, EndToEndMeanCentering) {
    Mat X_train = Mat::ones(5, 10, CV_32FC1) * 100;
    Mat XTrain_trans = X_train.t();
    
    // Calculate mean
    Mat sum_train = Mat::zeros(10, 1, CV_32FC1);
    for (int i = 0; i < 5; i++) {
        sum_train += XTrain_trans.col(i);
    }
    
    Mat mean_train;
    divide(sum_train, 5, mean_train);
    
    // Center
    for (int i = 0; i < 5; i++) {
        XTrain_trans.col(i) = XTrain_trans.col(i) - mean_train;
    }
    
    // Verify centered
    double max_val;
    minMaxLoc(abs(XTrain_trans), nullptr, &max_val);
    
    EXPECT_LT(max_val, 1e-5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
