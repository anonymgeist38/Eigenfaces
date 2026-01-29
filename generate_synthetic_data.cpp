#include <iostream>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {
    const int NO_OF_IMAGES = 2429;
    const int IMG_WIDTH = 19;
    const int IMG_HEIGHT = 19;
    
    // Create output directory structure
    string output_dir = "images/train/face";
    cout << "Creating synthetic face data..." << endl;
    
    // Initialize random number generator
    mt19937 gen(42); // seed for reproducibility
    uniform_int_distribution<> dis(0, 255);
    
    // Generate synthetic images
    for (int i = 0; i < NO_OF_IMAGES; i++) {
        Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);
        
        // Fill with random values
        for (int row = 0; row < IMG_HEIGHT; row++) {
            for (int col = 0; col < IMG_WIDTH; col++) {
                img.at<uchar>(row, col) = dis(gen);
            }
        }
        
        // Save as .pgm file
        string filename = cv::format("%s/face%05d.pgm", output_dir.c_str(), i);
        if (!imwrite(filename, img)) {
            cerr << "Failed to write: " << filename << endl;
            return 1;
        }
        
        if ((i + 1) % 500 == 0) {
            cout << "Generated " << (i + 1) << " images..." << endl;
        }
    }
    
    cout << "Successfully generated " << NO_OF_IMAGES << " synthetic face images in " << output_dir << endl;
    return 0;
}
