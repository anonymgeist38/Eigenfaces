#include <iostream>
#include <vector>
#include <algorithm>
#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QFont>
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

// Convert OpenCV Mat to QPixmap for display
QPixmap matToQPixmap(const Mat& mat) {
    if (mat.channels() == 1) {
        // Grayscale image
        QImage img((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return QPixmap::fromImage(img);
    } else {
        // Color image
        Mat rgbMat;
        cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
        QImage img((uchar*)rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888);
        return QPixmap::fromImage(img);
    }
}

int main(int argc, char *argv[]) {
    cout << "========================================" << endl;
    cout << "Eigenfaces Visualization with QCustomPlot" << endl;
    cout << "========================================" << endl << endl;

    // Load and prepare data
    cout << "[1] Loading face images..." << endl;
    vector<Mat> faces = read_faces();
    cout << "Loaded " << faces.size() << " images" << endl << endl;

    cout << "[2] Splitting data..." << endl;
    vector<Mat> test_set;
    vector<Mat> training_set = extract_train_test_set(faces, test_set);
    cout << "Training set: " << training_set.size() << endl << endl;

    cout << "[3] Constructing training matrix..." << endl;
    int dim = 19 * 19; // 361
    Mat X_train1(training_set.size(), dim, CV_8UC1);
    
    for (int index = 0; index < (int)training_set.size(); index++) {
        Mat imgFlat = training_set.at(index).reshape(1, 1);
        imgFlat.copyTo(X_train1.row(index));
    }
    cout << "Training matrix: " << X_train1.rows << "x" << X_train1.cols << endl << endl;

    cout << "[4] Preparing data for PCA..." << endl;
    Mat XTrain_trans = X_train1.t();
    XTrain_trans.convertTo(XTrain_trans, CV_32FC1);
    
    // Compute mean
    Mat mean_train(dim, 1, CV_32FC1, Scalar(0));
    for (int i = 0; i < (int)training_set.size(); i++) {
        mean_train += XTrain_trans.col(i);
    }
    mean_train = mean_train / (float)training_set.size();
    
    // Center data
    for (int i = 0; i < (int)training_set.size(); i++) {
        XTrain_trans.col(i) = XTrain_trans.col(i) - mean_train;
    }
    cout << "Data centered" << endl << endl;

    cout << "[5] Computing PCA decomposition..." << endl;
    Mat covariance_matrix;
    Mat meanMat;
    calcCovarMatrix(XTrain_trans, covariance_matrix, meanMat, COVAR_ROWS);
    covariance_matrix = covariance_matrix / training_set.size();
    
    Mat E, U, Vt;
    SVD::compute(covariance_matrix, E, U, Vt);
    Mat V = Vt.t();
    cout << "SVD computed - Eigenvalues: " << E.rows << endl;
    cout << "Eigenvectors: " << V.rows << "x" << V.cols << endl << endl;

    // Convert eigenvalues
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
    
    cout << "[6] Setting up Qt visualization..." << endl;
    QApplication app(argc, argv);

    // ============== WINDOW 1: Eigenfaces Grid ==============
    cout << "[7] Creating eigenfaces grid window..." << endl;
    QMainWindow mainWindow;
    mainWindow.setWindowTitle("Eigenfaces - Principal Components");
    mainWindow.resize(1400, 900);

    QWidget *centralWidget = new QWidget();
    mainWindow.setCentralWidget(centralWidget);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    QLabel *titleLabel = new QLabel("Principal Component Eigenvectors (Eigenfaces)");
    titleLabel->setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;");
    mainLayout->addWidget(titleLabel);

    // Create scroll area for eigenfaces
    QScrollArea *scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    mainLayout->addWidget(scrollArea);

    QWidget *gridWidget = new QWidget();
    QGridLayout *gridLayout = new QGridLayout(gridWidget);
    gridLayout->setSpacing(10);

    // Visualize top eigenfaces
    int num_to_visualize = 16;  // 4x4 grid
    int idx = 0;
    
    cout << "Creating " << num_to_visualize << " eigenface visualizations..." << endl;
    
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            if (idx < num_to_visualize) {
                // Extract eigenvector
                Mat eigenvector = V.row(idx).clone();
                if (eigenvector.depth() != CV_32F) {
                    eigenvector.convertTo(eigenvector, CV_32FC1);
                }
                
                // Reshape to 19x19
                Mat eigenface = eigenvector.reshape(1, 19);
                
                // Normalize to 0-255
                Mat dst;
                normalize(eigenface, dst, 0, 255, NORM_MINMAX, CV_8UC1);
                
                // Resize for display
                resize(dst, dst, Size(150, 150));
                
                // Create label with eigenface image and info
                QLabel *label = new QLabel();
                QPixmap pixmap = matToQPixmap(dst);
                label->setPixmap(pixmap);
                label->setAlignment(Qt::AlignCenter);
                
                // Calculate variance contribution
                float variance_ratio = 0.0f;
                for (int k = 0; k <= idx; k++) {
                    variance_ratio += eigenvalues_vec[k];
                }
                variance_ratio = (variance_ratio / sumOfEigenValues) * 100.0f;
                
                // Create cell widget
                QWidget *cellWidget = new QWidget();
                QVBoxLayout *cellLayout = new QVBoxLayout(cellWidget);
                
                QLabel *infoLabel = new QLabel();
                QString infoText = QString("Eigenface %1\nCumulative Var: %2%")
                                    .arg(idx)
                                    .arg(variance_ratio, 0, 'f', 1);
                infoLabel->setText(infoText);
                infoLabel->setStyleSheet("font-size: 9px; text-align: center;");
                infoLabel->setAlignment(Qt::AlignCenter);
                
                cellLayout->addWidget(label, 0, Qt::AlignCenter);
                cellLayout->addWidget(infoLabel);
                cellLayout->setContentsMargins(5, 5, 5, 5);
                
                gridLayout->addWidget(cellWidget, row, col);
                
                cout << "  ✓ Eigenface " << idx << " (cumulative variance: " 
                     << variance_ratio << "%)" << endl;
            }
            idx++;
        }
    }

    gridWidget->setLayout(gridLayout);
    scrollArea->setWidget(gridWidget);

    // Add info section at bottom
    QLabel *infoLabel = new QLabel();
    infoLabel->setText(QString("Total Eigenfaces: %1 | All variance explained at component: 361 | Sum of eigenvalues: %2")
                        .arg(V.rows)
                        .arg(sumOfEigenValues, 0, 'f', 2));
    infoLabel->setStyleSheet("background-color: #f0f0f0; padding: 8px; font-size: 10px;");
    mainLayout->addWidget(infoLabel);

    // ============== WINDOW 2: Mean Face ==============
    cout << "[8] Creating mean face visualization..." << endl;
    QMainWindow meanWindow;
    meanWindow.setWindowTitle("Mean Face");
    meanWindow.resize(500, 550);

    QWidget *meanWidget = new QWidget();
    QVBoxLayout *meanLayout = new QVBoxLayout(meanWidget);
    meanWindow.setCentralWidget(meanWidget);

    Mat meanImg = mean_train.reshape(1, 19);
    Mat meanImg_vis;
    normalize(meanImg, meanImg_vis, 0, 255, NORM_MINMAX, CV_8UC1);
    resize(meanImg_vis, meanImg_vis, Size(300, 300));

    QLabel *meanPixmapLabel = new QLabel();
    meanPixmapLabel->setPixmap(matToQPixmap(meanImg_vis));
    meanPixmapLabel->setAlignment(Qt::AlignCenter);

    QLabel *meanTitleLabel = new QLabel("Mean Face");
    meanTitleLabel->setStyleSheet("font-weight: bold; font-size: 12px; text-align: center;");
    meanTitleLabel->setAlignment(Qt::AlignCenter);

    QLabel *meanDescLabel = new QLabel("Average of all 2186 training images");
    meanDescLabel->setStyleSheet("font-size: 10px; text-align: center;");
    meanDescLabel->setAlignment(Qt::AlignCenter);

    meanLayout->addWidget(meanTitleLabel);
    meanLayout->addWidget(meanDescLabel);
    meanLayout->addWidget(meanPixmapLabel);
    meanLayout->addStretch();

    meanWindow.show();

    // ============== WINDOW 3: Eigenvalue Spectrum ==============
    cout << "[9] Creating eigenvalue spectrum plot..." << endl;
    QMainWindow spectrumWindow;
    spectrumWindow.setWindowTitle("Eigenvalue Spectrum");
    spectrumWindow.resize(1000, 600);

    QWidget *specWidget = new QWidget();
    QVBoxLayout *specLayout = new QVBoxLayout(specWidget);
    spectrumWindow.setCentralWidget(specWidget);

    QLabel *specTitle = new QLabel("Eigenvalue Spectrum (Log Scale)");
    specTitle->setStyleSheet("font-weight: bold; font-size: 12px;");
    specLayout->addWidget(specTitle);

    // Create image for spectrum
    QImage specImage(800, 500, QImage::Format_RGB32);
    specImage.fill(Qt::white);

    QPainter painter(&specImage);
    painter.setRenderHint(QPainter::Antialiasing);

    // Find max eigenvalue for scaling
    float maxEig = *max_element(eigenvalues_vec.begin(), eigenvalues_vec.end());
    
    // Draw axis
    painter.drawLine(50, 450, 750, 450);  // x-axis
    painter.drawLine(50, 450, 50, 50);    // y-axis

    // Draw spectrum bars
    float barWidth = 700.0f / eigenvalues_vec.size();
    for (int i = 0; i < (int)eigenvalues_vec.size(); i++) {
        float height = (eigenvalues_vec[i] / maxEig) * 400;
        float x = 50 + i * barWidth;
        
        if (height > 0) {
            QRect bar(x, 450 - height, barWidth - 1, height);
            painter.fillRect(bar, QColor(70, 130, 180));
        }
    }

    // Labels
    painter.setFont(QFont("Arial", 10));
    painter.drawText(400, 480, "Principal Component Index");
    painter.save();
    painter.translate(20, 250);
    painter.rotate(-90);
    painter.drawText(0, 0, "Eigenvalue (log scale)");
    painter.restore();

    QLabel *specLabel = new QLabel();
    specLabel->setPixmap(QPixmap::fromImage(specImage));
    specLabel->setAlignment(Qt::AlignCenter);
    specLayout->addWidget(specLabel);

    // Save spectrum image
    specImage.save("images/train/eigenVecVis/spectrum_plot.png");
    cout << "  ✓ Spectrum plot saved" << endl;

    spectrumWindow.show();

    // ============== WINDOW 4: Cumulative Variance ==============
    cout << "[10] Creating cumulative variance plot..." << endl;
    QMainWindow varianceWindow;
    varianceWindow.setWindowTitle("Cumulative Variance Explained");
    varianceWindow.resize(1000, 600);

    QWidget *varWidget = new QWidget();
    QVBoxLayout *varLayout = new QVBoxLayout(varWidget);
    varianceWindow.setCentralWidget(varWidget);

    QLabel *varTitle = new QLabel("Cumulative Variance Explained");
    varTitle->setStyleSheet("font-weight: bold; font-size: 12px;");
    varLayout->addWidget(varTitle);

    // Create image for variance plot
    QImage varImage(800, 500, QImage::Format_RGB32);
    varImage.fill(Qt::white);

    QPainter varPainter(&varImage);
    varPainter.setRenderHint(QPainter::Antialiasing);

    // Draw axis
    varPainter.drawLine(50, 450, 750, 450);  // x-axis
    varPainter.drawLine(50, 450, 50, 50);    // y-axis

    // Draw variance curve
    varPainter.setPen(QPen(Qt::darkGreen, 2));
    float cumsum = 0.0f;
    float prevX = 50, prevY = 450;
    
    for (int i = 0; i < (int)eigenvalues_vec.size(); i++) {
        cumsum += eigenvalues_vec[i];
        float varRatio = (cumsum / sumOfEigenValues) * 100.0f;
        float x = 50 + (i / (float)eigenvalues_vec.size()) * 700;
        float y = 450 - (varRatio / 100.0f) * 400;
        
        varPainter.drawLine(prevX, prevY, x, y);
        prevX = x;
        prevY = y;
    }

    // Draw 90% threshold line
    varPainter.setPen(QPen(Qt::red, 2, Qt::DashLine));
    float y90 = 450 - 0.9f * 400;
    varPainter.drawLine(50, y90, 750, y90);
    varPainter.drawText(760, y90, "90%");

    // Labels
    varPainter.setPen(Qt::black);
    varPainter.setFont(QFont("Arial", 10));
    varPainter.drawText(400, 480, "Principal Component Index");
    varPainter.save();
    varPainter.translate(20, 250);
    varPainter.rotate(-90);
    varPainter.drawText(0, 0, "Cumulative Variance (%)");
    varPainter.restore();

    QLabel *varLabel = new QLabel();
    varLabel->setPixmap(QPixmap::fromImage(varImage));
    varLabel->setAlignment(Qt::AlignCenter);
    varLayout->addWidget(varLabel);

    // Save variance image
    varImage.save("images/train/eigenVecVis/variance_plot.png");
    cout << "  ✓ Cumulative variance plot saved" << endl;

    varianceWindow.show();

    cout << endl << "========================================" << endl;
    cout << "Visualizations created successfully!" << endl;
    cout << "========================================" << endl;
    cout << "Windows:" << endl;
    cout << "  1. Eigenfaces Grid (16 principal components)" << endl;
    cout << "  2. Mean Face (average training image)" << endl;
    cout << "  3. Eigenvalue Spectrum" << endl;
    cout << "  4. Cumulative Variance Explained" << endl;
    cout << endl << "Plots saved to: images/train/eigenVecVis/" << endl;
    cout << endl;

    mainWindow.show();

    return app.exec();
}
