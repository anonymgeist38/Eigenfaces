cmake_minimum_required(VERSION 3.10)
project(Eigenfaces)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Qt6 COMPONENTS Core Gui Widgets PrintSupport REQUIRED)

# Eigenfaces visualization with QCustomPlot
add_executable(visualize_eigenfaces_qt visualize_eigenfaces_qt.cpp)
target_link_libraries(visualize_eigenfaces_qt PUBLIC ${OpenCV_LIBS} Qt6::Core Qt6::Gui Qt6::Widgets)

# Data generation utility
add_executable(generate_data generate_synthetic_data.cpp)
target_link_libraries(generate_data PUBLIC ${OpenCV_LIBS})

# Eigenfaces visualization
add_executable(visualize_eigenfaces visualize_eigenfaces.cpp)
target_link_libraries(visualize_eigenfaces PUBLIC ${OpenCV_LIBS})


# Main executable with Qt GUI and QCustomPlot
add_executable(eigenfaces eigenfaces.cpp qcustomplot.cpp qcustomplot.h)
target_include_directories(eigenfaces PRIVATE /opt/homebrew/include)
target_link_libraries(eigenfaces PUBLIC ${OpenCV_LIBS} Qt6::Core Qt6::Gui Qt6::Widgets Qt6::PrintSupport)

# Main executable with GUI - commented out until QCustomPlot is available
# add_executable(eigenfaces_gui eigenfaces.cpp)
# target_link_libraries(eigenfaces_gui PUBLIC ${OpenCV_LIBS} Qt6::Core Qt6::Gui Qt6::Widgets)

# Test executable
enable_testing()
add_executable(test_eigenfaces test_eigenfaces.cpp)
target_link_libraries(test_eigenfaces PUBLIC GTest::gtest GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME EigenfacesTests COMMAND test_eigenfaces)
include_directories(${OpenCV_INCLUDE_DIRS})

