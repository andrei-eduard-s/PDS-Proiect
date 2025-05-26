#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <string>
#include <numeric>
#include <arm_neon.h> // NEON intrinseci

// Numara tranzitiile 0->1 in vectorul de vecini (folosit in Zhang-Suen)
int count_transitions(const std::vector<int>& neighbors) {
    int transitions = 0;
    for (size_t k = 0; k < neighbors.size(); k++) {
        if (neighbors[k] == 0 && neighbors[(k + 1) % neighbors.size()] == 1) {
            transitions++;
        }
    }
    return transitions;
}

// Obtine cei 8 vecini folosind NEON intrinseci (accelereaza accesul la pixeli vecini)
std::vector<int> get_neighbors_neon(const cv::Mat& img, int i, int j) {
    // Colectam pixelii vecini in ordinea corecta (in jurul pixelului curent)
    uint8x8_t neighbors_raw = vcreate_u8(
        ((uint64_t)img.at<uchar>(i - 1, j) << 0)     |
        ((uint64_t)img.at<uchar>(i - 1, j + 1) << 8) |
        ((uint64_t)img.at<uchar>(i, j + 1) << 16)    |
        ((uint64_t)img.at<uchar>(i + 1, j + 1) << 24)|
        ((uint64_t)img.at<uchar>(i + 1, j) << 32)    |
        ((uint64_t)img.at<uchar>(i + 1, j - 1) << 40)|
        ((uint64_t)img.at<uchar>(i, j - 1) << 48)    |
        ((uint64_t)img.at<uchar>(i - 1, j - 1) << 56)
    );

    // Prag de 0 pentru a obtine 0 sau 1 (pixels binari)
    uint8x8_t thresholded = vcgt_u8(neighbors_raw, vdup_n_u8(0));

    std::vector<int> neighbors(8);
    for (int k = 0; k < 8; k++) {
        neighbors[k] = (thresholded[k] != 0) ? 1 : 0;
    }
    return neighbors;
}

// Zhang-Suen thinning - 500 iteratii (sau pana la stabilizare)
// Foloseste get_neighbors_neon pentru acces rapid la vecini (accelerare NEON)
cv::Mat zhang_suen_thinning(const cv::Mat& binary_image, int max_iter = 500) {
    cv::Mat thinned;
    binary_image.copyTo(thinned);
    thinned /= 255;  // Convertim imaginea la valori binare 0/1

    int iteration = 0;
    cv::Mat prev = cv::Mat::zeros(thinned.size(), CV_8UC1);

    while (true) {
        cv::Mat marker = cv::Mat::zeros(thinned.size(), CV_8UC1);

        // Sub-iteratia 1
        for (int i = 1; i < thinned.rows - 1; i++) {
            for (int j = 1; j < thinned.cols - 1; j++) {
                if (thinned.at<uchar>(i, j) != 1) continue;

                std::vector<int> neighbors = get_neighbors_neon(thinned, i, j);
                int count = std::accumulate(neighbors.begin(), neighbors.end(), 0);
                int transitions = count_transitions(neighbors);

                if (count >= 2 && count <= 6 && transitions == 1 &&
                    neighbors[0] * neighbors[2] * neighbors[4] == 0 &&
                    neighbors[2] * neighbors[4] * neighbors[6] == 0) {
                    marker.at<uchar>(i, j) = 1;
                }
            }
        }
        thinned -= marker;

        // Sub-iteratia 2
        marker = cv::Mat::zeros(thinned.size(), CV_8UC1);
        for (int i = 1; i < thinned.rows - 1; i++) {
            for (int j = 1; j < thinned.cols - 1; j++) {
                if (thinned.at<uchar>(i, j) != 1) continue;

                std::vector<int> neighbors = get_neighbors_neon(thinned, i, j);
                int count = std::accumulate(neighbors.begin(), neighbors.end(), 0);
                int transitions = count_transitions(neighbors);

                if (count >= 2 && count <= 6 && transitions == 1 &&
                    neighbors[0] * neighbors[2] * neighbors[6] == 0 &&
                    neighbors[0] * neighbors[4] * neighbors[6] == 0) {
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

    thinned *= 255; // Reconversie la 0/255 pentru vizualizare
    return thinned;
}

// Structura pentru minutii fingerprint
struct Minutia {
    cv::Point position;
    std::string type;  // "ending" sau "bifurcation"
};

// Extrage minutii folosind vecinii NEON si algoritmul standard
std::vector<Minutia> extract_minutiae(const cv::Mat& thinned) {
    std::vector<Minutia> minutiae;
    for (int i = 1; i < thinned.rows - 1; i++) {
        for (int j = 1; j < thinned.cols - 1; j++) {
            if (thinned.at<uchar>(i, j) == 255) {
                std::vector<int> neighbors = get_neighbors_neon(thinned, i, j);
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

// Functie care proceseaza si masoara timpul total pentru thinning + extragerea minutiei pe 500 iteratii
void process_and_measure(const cv::Mat& binary_image, const std::string& log_file_path, 
                         const std::string& output_img_path) {
    using clock = std::chrono::high_resolution_clock;

    cv::Mat last_thinned;
    std::vector<Minutia> last_minutiae;

    auto start = clock::now();

    for (int k = 0; k < 500; ++k) {
        last_thinned = zhang_suen_thinning(binary_image, 20);
        last_minutiae = extract_minutiae(last_thinned);
    }

    auto end = clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::ofstream log_file(log_file_path, std::ios::app);
    if (log_file.is_open()) {
        log_file << "Execution time (Zhang-Suen + minutiae, 500 runs): " << elapsed_seconds.count() << " seconds\n";
        log_file.close();
    } else {
        std::cerr << "Cannot open log file: " << log_file_path << std::endl;
    }

    // Salvam doar ultima imagine procesata
    cv::Mat minutiae_img;
    cv::cvtColor(last_thinned, minutiae_img, cv::COLOR_GRAY2BGR);

    for (const auto& m : last_minutiae) {
        cv::Scalar color = (m.type == "ending") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::circle(minutiae_img, m.position, 2, color, -1);
    }
    cv::imwrite(output_img_path, minutiae_img);

    std::cout << "Thinning + minutiae extraction done in: " << elapsed_seconds.count() << " seconds for 500 runs\n";
    std::cout << "Result image saved to: " << output_img_path << std::endl;
    std::cout << "Log written to: " << log_file_path << std::endl;
}


int main() {
    // Calea imaginilor si fisierelor
    std::string input_path = "/home/user/PDS/neon/input_images/100__M_Left_index_finger.BMP";
    std::string output_path = "/home/user/PDS/neon/output_images/result.png";
    std::string log_path = "/home/user/PDS/neon/logs/runtime_log.txt";

    // Citim imaginea in grayscale
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error reading image\n";
        return -1;
    }

    // Preprocesare: CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(img, enhanced);

    // Binarizare adaptiva
    cv::Mat binary;
    cv::adaptiveThreshold(enhanced, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 11, 2);

    // Procesam si masuram timpul DOAR pentru thinning + extragerea minutiei
    process_and_measure(binary, log_path, output_path);

    return 0;
}
