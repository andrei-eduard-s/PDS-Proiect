#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <chrono>
#include <filesystem>
#include <string>
#include <numeric>

// Functie de numarare a tranzitiilor 0->1 in vecinatatea pixelului
int count_transitions(const std::vector<int>& neighbors) {
    int transitions = 0;
    for (size_t k = 0; k < neighbors.size(); k++) {
        if (neighbors[k] == 0 && neighbors[(k + 1) % neighbors.size()] == 1) {
            transitions++;
        }
    }
    return transitions;
}

// Zhang-Suen thinning
cv::Mat zhang_suen_thinning(const cv::Mat& binary_image, int max_iter = 20) {
    cv::Mat thinned;
    binary_image.copyTo(thinned);
    thinned /= 255;  // Convert to 0/1 for processing

    int iteration = 0;
    cv::Mat prev = cv::Mat::zeros(thinned.size(), CV_8UC1);
    while (true) {
        cv::Mat marker = cv::Mat::zeros(thinned.size(), CV_8UC1);
        for (int i = 1; i < thinned.rows - 1; i++) {
            for (int j = 1; j < thinned.cols - 1; j++) {
                uchar p = thinned.at<uchar>(i, j);
                if (p != 1) continue;

                std::vector<int> N = {
                    thinned.at<uchar>(i - 1, j),     // P2
                    thinned.at<uchar>(i - 1, j + 1), // P3
                    thinned.at<uchar>(i, j + 1),     // P4
                    thinned.at<uchar>(i + 1, j + 1), // P5
                    thinned.at<uchar>(i + 1, j),     // P6
                    thinned.at<uchar>(i + 1, j - 1), // P7
                    thinned.at<uchar>(i, j - 1),     // P8
                    thinned.at<uchar>(i - 1, j - 1)  // P9
                };

                int count = std::accumulate(N.begin(), N.end(), 0);
                int transitions = count_transitions(N);

                if (count >= 2 && count <= 6 && transitions == 1 &&
                    N[0] * N[2] * N[4] == 0 &&
                    N[2] * N[4] * N[6] == 0) {
                    marker.at<uchar>(i, j) = 1;
                }
            }
        }

        thinned -= marker;

        marker = cv::Mat::zeros(thinned.size(), CV_8UC1);
        for (int i = 1; i < thinned.rows - 1; i++) {
            for (int j = 1; j < thinned.cols - 1; j++) {
                uchar p = thinned.at<uchar>(i, j);
                if (p != 1) continue;

                std::vector<int> N = {
                    thinned.at<uchar>(i - 1, j),     // P2
                    thinned.at<uchar>(i - 1, j + 1), // P3
                    thinned.at<uchar>(i, j + 1),     // P4
                    thinned.at<uchar>(i + 1, j + 1), // P5
                    thinned.at<uchar>(i + 1, j),     // P6
                    thinned.at<uchar>(i + 1, j - 1), // P7
                    thinned.at<uchar>(i, j - 1),     // P8
                    thinned.at<uchar>(i - 1, j - 1)  // P9
                };

                int count = std::accumulate(N.begin(), N.end(), 0);
                int transitions = count_transitions(N);

                if (count >= 2 && count <= 6 && transitions == 1 &&
                    N[0] * N[2] * N[6] == 0 &&
                    N[0] * N[4] * N[6] == 0) {
                    marker.at<uchar>(i, j) = 1;
                }
            }
        }

        thinned -= marker;

        if (cv::countNonZero(thinned != prev) == 0 || iteration++ >= max_iter) {
            break;
        }

        thinned.copyTo(prev);
    }

    thinned *= 255;  // Convert back to 0/255
    return thinned;
}

// Structura pentru minutiae
struct Minutia {
    cv::Point position;
    std::string type; // "ending" or "bifurcation"
};

// Functie de extragere minutiae
std::vector<Minutia> extract_minutiae(const cv::Mat& thinned) {
    std::vector<Minutia> minutiae;
    for (int i = 1; i < thinned.rows - 1; i++) {
        for (int j = 1; j < thinned.cols - 1; j++) {
            if (thinned.at<uchar>(i, j) == 255) {
                std::vector<int> neighbors = {
                    thinned.at<uchar>(i - 1, j), thinned.at<uchar>(i - 1, j + 1),
                    thinned.at<uchar>(i, j + 1), thinned.at<uchar>(i + 1, j + 1),
                    thinned.at<uchar>(i + 1, j), thinned.at<uchar>(i + 1, j - 1),
                    thinned.at<uchar>(i, j - 1), thinned.at<uchar>(i - 1, j - 1)
                };

                for (auto& n : neighbors) n = (n == 255) ? 1 : 0;
                int transitions = count_transitions(neighbors);
                int sum_neighbors = std::accumulate(neighbors.begin(), neighbors.end(), 0);

                if (transitions == 1 && sum_neighbors >= 1) {
                    minutiae.push_back({cv::Point(j, i), "ending"});
                } else if (transitions == 3) {
                    minutiae.push_back({cv::Point(j, i), "bifurcation"});
                }
            }
        }
    }
    return minutiae;
}

// Functia principala de procesare a unei imagini
void process_fingerprint_image(const std::string& image_path, const std::string& save_path) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Eroare la citirea imaginii: " << image_path << std::endl;
        return;
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(img, enhanced);

    cv::Mat binary;
    cv::adaptiveThreshold(enhanced, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    cv::Mat thinned = zhang_suen_thinning(binary);

    std::vector<Minutia> minutiae = extract_minutiae(thinned);

    // Vizualizare rezultat
    cv::Mat minutiae_img;
    cv::cvtColor(thinned, minutiae_img, cv::COLOR_GRAY2BGR);
    for (const auto& m : minutiae) {
        cv::Scalar color = (m.type == "ending") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255); // verde / rosu
        cv::circle(minutiae_img, m.position, 2, color, -1);
    }

    cv::imwrite(save_path, minutiae_img);
}

// Main
int main() {
    using namespace std::chrono;

    std::string image_folder = "/home/user/PDS/neon/input_images/";
    std::string output_folder = "/home/user/PDS/neon/output_images/";
    std::string log_folder = "/home/user/PDS/neon/logs/";

    // Verifica si creaza folderele
    std::filesystem::create_directories(image_folder);
    std::filesystem::create_directories(output_folder);
    std::filesystem::create_directories(log_folder); // Creaza folderul pentru logs

    // Masurarea timpului de procesare
    auto start = high_resolution_clock::now();

    int processed_images = 0;

    // Proceseaza primele 20 de imagini
    for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
        if (processed_images >= 20) break;

        std::string image_path = entry.path().string();
        std::string save_path = output_folder + "processed_" + std::to_string(processed_images + 1) + ".png";

        process_fingerprint_image(image_path, save_path);
        std::cout << "Processed image: " << image_path << " -> " << save_path << std::endl;

        processed_images++;
    }

    // Masurarea timpului final
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // Afisarea timpului total in consola
    std::cout << "Total processing time: " 
              << duration.count() / 1000.0 << " seconds." << std::endl;

    // Salvarea timpului in fisierul log
    std::ofstream log_file(log_folder + "process_time.txt");
    log_file << "Total processing time: " 
             << duration.count() / 1000.0 << " seconds." << std::endl;

    return 0;
}