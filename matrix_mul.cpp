#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <clocale>
#include <omp.h>

using namespace std;
using namespace chrono;

typedef long long int big_int;

class MatrixMultiplier {
private:
    vector<vector<big_int>> matrixA;
    vector<vector<big_int>> matrixB;
    vector<vector<big_int>> matrixC;
    int size;
    int num_threads;

public:
    MatrixMultiplier(int threads) : size(0), num_threads(threads) {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
    }

    bool readMatrices(const string& filenameA, const string& filenameB) {
        if (!readMatrix(filenameA, matrixA) || !readMatrix(filenameB, matrixB)) {
            return false;
        }

        if (matrixA.size() != matrixB.size() || matrixA[0].size() != matrixB[0].size()) {
            cerr << "The matrices must be the same size!" << endl;
            return false;
        }

        size = matrixA.size();
        // Initialize result matrix with zeros
        matrixC.assign(size, vector<big_int>(size, 0));
        return true;
    }

    bool readMatrix(const string& filename, vector<vector<big_int>>& matrix) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            return false;
        }

        matrix.clear();
        string line;
        while (getline(file, line)) {
            vector<big_int> row;
            stringstream ss(line);
            big_int value;
            while (ss >> value) {
                row.push_back(value);
            }
            if (!row.empty()) {
                matrix.push_back(row);
            }
        }
        file.close();

        if (matrix.empty()) return false;
        int rows = matrix.size();
        for (const auto& row : matrix) {
            if (row.size() != rows) {
                cerr << "The matrix is not square!" << endl;
                return false;
            }
        }
        return true;
    }

    double multiplyMatrixesParallel() {
        auto start = high_resolution_clock::now();

#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++) {
                big_int temp = matrixA[i][k];
                for (int j = 0; j < size; j++) {
                    matrixC[i][j] += temp * matrixB[k][j];
                }
            }
        }

        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;
        return elapsed.count();
    }

    void writeResult(const string& filename) {
        ofstream file(filename);
        if (!file.is_open()) return;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file << matrixC[i][j] << (j == size - 1 ? "" : " ");
            }
            file << "\n";
        }
        file.close();
    }
};

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");

    string fileA = "matrix_a.txt";
    string fileB = "matrix_b.txt";
    string fileOut = "matrix_c.txt";

    int num_threads = omp_get_max_threads();

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-a" && i + 1 < argc) fileA = argv[++i];
        else if (arg == "-b" && i + 1 < argc) fileB = argv[++i];
        else if (arg == "-o" && i + 1 < argc) fileOut = argv[++i];
        else if (arg == "-t" && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]\n"
                << "  -a <file>         matrix input file A\n"
                << "  -b <file>         matrix input file B\n"
                << "  -o <file>         result matrix file\n"
                << "  -t <threads>      number of threads\n"
                << "  --help            show this certificate\n";
            return 0;
        }
    }

    cout << "Running with " << num_threads << " threads." << endl;

    MatrixMultiplier multiplier(num_threads);

    if (!multiplier.readMatrices(fileA, fileB)) {
        cerr << "Error reading matrices!" << endl;
        return 1;
    }

    double time_taken = multiplier.multiplyMatrixesParallel();
    cout << "Calculation finished in " << fixed << setprecision(4) << time_taken << " seconds." << endl;

    multiplier.writeResult(fileOut);

    return 0;
}