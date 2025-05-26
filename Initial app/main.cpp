#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <numeric>

// Functie care numara tranzitiile 0 -> 1 in jurul unui pixel
int count_transitions(const std::vector<int>& neighbors) {
    int transitions = 0;
    for (size_t k = 0; k < neighbors.size(); k++) {
        if (neighbors[k] == 0 && neighbors[(k + 1) % neighbors.size()] == 1) {
            transitions++;
        }
    }
    return transitions;
}

// Functie care extrage cei 8 vecini ai unui pixel
std::vector<int> get_neighbors(const cv::Mat& img, int i, int j) {
    std::vector<int> neighbors = {
        img.at<uchar>(i - 1, j) > 0 ? 1 : 0,
        img.at<uchar>(i - 1, j + 1) > 0 ? 1 : 0,
        img.at<uchar>(i, j + 1) > 0 ? 1 : 0,
        img.at<uchar>(i + 1, j + 1) > 0 ? 1 : 0,
        img.at<uchar>(i + 1, j) > 0 ? 1 : 0,
        img.at<uchar>(i + 1, j - 1) > 0 ? 1 : 0,
        img.at<uchar>(i, j - 1) > 0 ? 1 : 0,
        img.at<uchar>(i - 1, j - 1) > 0 ? 1 : 0
    };
    return neighbors;
}

// Algoritmul Zhang-Suen pentru subtierea imaginii (fara NEON)
cv::Mat zhang_suen_thinning_plain(const cv::Mat& binary_image, int max_iter = 20) {
    cv::Mat thinned;
    binary_image.copyTo(thinned);
    thinned /= 255;  // Convertim imaginea binara la valori 0 si 1

    int iteration = 0;
    cv::Mat prev = cv::Mat::zeros(thinned.size(), CV_8UC1);

    while (true) {
        cv::Mat marker = cv::Mat::zeros(thinned.size(), CV_8UC1);

        // Prima subetapa
        for (int i = 1; i < thinned.rows - 1; i++) {
            for (int j = 1; j < thinned.cols - 1; j++) {
                if (thinned.at<uchar>(i, j) != 1) continue;

                std::vector<int> neighbors = get_neighbors(thinned, i, j);
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

        // A doua subetapa
        marker = cv::Mat::zeros(thinned.size(), CV_8UC1);
        for (int i = 1; i < thinned.rows - 1; i++) {
            for (int j = 1; j < thinned.cols - 1; j++) {
                if (thinned.at<uchar>(i, j) != 1) continue;

                std::vector<int> neighbors = get_neighbors(thinned, i, j);
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

        // Verificam daca imaginea s-a modificat sau am atins numarul maxim de iteratii
        if (cv::countNonZero(thinned != prev) == 0 || iteration++ >= max_iter) {
            break;
        }
        thinned.copyTo(prev);
    }

    thinned *= 255;  // Convertim imaginea inapoi la 0/255
    return thinned;
}

// Structura pentru reprezentarea unei minutii (terminatie sau bifurcatie)
struct Minutia {
    cv::Point position;
    std::string type;
};

// Functie pentru extragerea minutiei dintr-o imagine subtire
std::vector<Minutia> extract_minutiae_plain(const cv::Mat& thinned) {
    std::vector<Minutia> minutiae;
    for (int i = 1; i < thinned.rows - 1; i++) {
        for (int j = 1; j < thinned.cols - 1; j++) {
            if (thinned.at<uchar>(i, j) == 255) {
                std::vector<int> neighbors = get_neighbors(thinned, i, j);
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

// Functie care proceseaza imaginea de 500 de ori si masoara timpul total
void process_plain(const cv::Mat& binary_image, const std::string& log_file_path,
                   const std::string& output_img_path) {
    using clock = std::chrono::high_resolution_clock;

    cv::Mat last_thinned;
    std::vector<Minutia> last_minutiae;

    auto start = clock::now();

    // Rulam algoritmul de 500 de ori
    for (int k = 0; k < 500; ++k) {
        last_thinned = zhang_suen_thinning_plain(binary_image, 20);
        last_minutiae = extract_minutiae_plain(last_thinned);
    }

    auto end = clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Salvam timpul de rulare in fisier log
    std::ofstream log_file(log_file_path, std::ios::app);
    if (log_file.is_open()) {
        log_file << "Execution time (plain Zhang-Suen + minutiae, 500 runs): "
                 << elapsed.count() << " seconds\n";
        log_file.close();
    } else {
        std::cerr << "Cannot open log file: " << log_file_path << std::endl;
    }

    // Desenam minutia pe imagine si o salvam
    cv::Mat output_img;
    cv::cvtColor(last_thinned, output_img, cv::COLOR_GRAY2BGR);
    for (const auto& m : last_minutiae) {
        cv::Scalar color = (m.type == "ending") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::circle(output_img, m.position, 2, color, -1);
    }
    cv::imwrite(output_img_path, output_img);

    std::cout << "Execution time (plain, 500 runs): " << elapsed.count() << " seconds\n";
    std::cout << "Saved image to: " << output_img_path << "\n";
}

int main() {
    // Cale imagine intrare, iesire, log
    std::string input_path = "/home/user/PDS/non_neon/input_images/100__M_Left_index_finger.BMP";
    std::string output_path = "/home/user/PDS/non_neon/output_images/result_plain.png";
    std::string log_path = "/home/user/PDS/non_neon/logs/runtime_log_plain.txt";

    // Citim imaginea grayscale
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image\n";
        return -1;
    }

    // Aplicam CLAHE pentru imbunatatirea contrastului
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(img, enhanced);

    // Binarizam imaginea
    cv::Mat binary;
    cv::adaptiveThreshold(enhanced, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 11, 2);

    // Procesam imaginea si masuram timpul pentru 500 iteratii
    process_plain(binary, log_path, output_path);

    return 0;
}
