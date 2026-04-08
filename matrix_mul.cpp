// matrix_multiply.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <clocale>

using namespace std;
using namespace chrono;

typedef long long int big_int;

class MatrixMultiplier {
private:
    vector<vector<big_int>> matrixA;
    vector<vector<big_int>> matrixB;
    vector<vector<big_int>> matrixC;
    int size;

public:
    MatrixMultiplier() : size(0) {}

    bool readMatrices(const string& filenameA, const string& filenameB) {
        if (!readMatrix(filenameA, matrixA) || !readMatrix(filenameB, matrixB)) {
            return false;
        }

        if (matrixA.size() != matrixB.size() || matrixA[0].size() != matrixB[0].size()) {
            cerr << "The matrices must be the same size!" << endl;
            return false;
        }

        size = matrixA.size();
        matrixC.resize(size, vector<big_int>(size, 0));
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

        int rows = matrix.size();
        for (const auto& row : matrix) {
            if (row.size() != rows) {
                cerr << "The matrix is ​​not square!" << endl;
                return false;
            }
        }

        return true;
    }

    double multiplyMatrixes() {
        auto start = high_resolution_clock::now();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrixC[i][j] = 0;
                for (int k = 0; k < size; k++) {
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;
        return elapsed.count();
    }

    void writeResult(const string& filename) {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to create result file: " << filename << endl;
            return;
        }

        file << fixed << setprecision(6);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file << matrixC[i][j];
                if (j < size - 1) file << " ";
            }
            file << endl;
        }

        file.close();
        cout << "The result is written to a file: " << filename << endl;
    }

    void printMatrix(const vector<vector<big_int>>& matrix, const string& name) {
        cout << "Matrix " << name << " (" << size << "x" << size << "):" << endl;
        for (int i = 0; i < min(size, 5); i++) {
            for (int j = 0; j < min(size, 5); j++) {
                cout << setw(10) << setprecision(4) << matrix[i][j] << " ";
            }
            if (size > 5) cout << "...";
            cout << endl;
        }
        if (size > 5) cout << "..." << endl;
    }

    int getSize() const { return size; }

    vector<vector<big_int>> getMatrixA() {
        return matrixA;
    }
    vector<vector<big_int>> getMatrixB() {
        return matrixB;
    }

    vector<vector<big_int>> getMatrixC() {
        return matrixC;
    }

    big_int getOperationsCount() const {
        return 2LL * size * size * size;
    }
};

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    string fileA, fileB, fileOut;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-a" && i + 1 < argc) fileA = argv[++i];
        else if (arg == "-b" && i + 1 < argc) fileB = argv[++i];
        else if (arg == "-o" && i + 1 < argc) fileOut = argv[++i];
        else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]\n"
                << "  -a <file>    matrix input file A\n"
                << "  -b <file>    matrix input file B\n"
                << "  -o <file>    result matrix file\n"
                << "  --help        show this certificate\n";
            return 0;
        }
    }

    if (fileA.empty()) fileA = "matrix_a.txt";
    if (fileB.empty()) fileB = "matrix_b.txt";
    if (fileOut.empty()) fileOut = "matrix_c.txt";

    MatrixMultiplier multiplier;

    if (!multiplier.readMatrices(fileA, fileB)) {
        cerr << "Error reading matrices!" << endl;
        return 1;
    }

    double elapsed;

    elapsed = multiplier.multiplyMatrixes();

    multiplier.writeResult(fileOut);

    return 0;
}