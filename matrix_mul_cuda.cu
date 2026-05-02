#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

typedef long long int big_int;

__global__ void matrixMulKernel(const big_int* A, const big_int* B, big_int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        big_int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

class MatrixMultiplier {
private:
    big_int* h_A, * h_B, * h_C;
    int size;
    int block_size;

public:
    MatrixMultiplier(int bs) : h_A(nullptr), h_B(nullptr), h_C(nullptr), size(0), block_size(bs) {}

    ~MatrixMultiplier() {
        if (h_A) delete[] h_A;
        if (h_B) delete[] h_B;
        if (h_C) delete[] h_C;
    }

    bool readMatrices(const string& filenameA, const string& filenameB) {
        int sizeA, sizeB;
        if (!readMatrix(filenameA, h_A, sizeA) || !readMatrix(filenameB, h_B, sizeB)) return false;
        if (sizeA != sizeB) {
            cerr << "Error: Matrices must have same dimensions!" << endl;
            return false;
        }
        size = sizeA;
        h_C = new big_int[size * size];
        return true;
    }

    bool readMatrix(const string& filename, big_int*& matrix, int& outSize) {
        ifstream file(filename);
        if (!file.is_open()) return false;

        vector<big_int> values;
        string line;
        int rows = 0;
        while (getline(file, line)) {
            stringstream ss(line);
            big_int val;
            int cols = 0;
            while (ss >> val) {
                values.push_back(val);
                cols++;
            }
            if (cols > 0) rows++;
        }
        outSize = rows;
        matrix = new big_int[values.size()];
        for (size_t i = 0; i < values.size(); i++) matrix[i] = values[i];
        return true;
    }

    double multiplyCUDA() {
        size_t bytes = size * size * sizeof(big_int);
        big_int* d_A, * d_B, * d_C;

        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

        auto start = high_resolution_clock::now();

        matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size);
        cudaDeviceSynchronize();

        auto end = high_resolution_clock::now();

        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return duration<double>(end - start).count();
    }

    void writeResult(const string& filename) {
        ofstream file(filename);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file << h_C[i * size + j] << (j == size - 1 ? "" : " ");
            }
            file << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    string fileA = "matrix_a.txt", fileB = "matrix_b.txt", fileOut = "matrix_c.txt";
    int block_size = 16;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-a") fileA = argv[++i];
        else if (arg == "-b") fileB = argv[++i];
        else if (arg == "-o") fileOut = argv[++i];
        else if (arg == "-bs") block_size = atoi(argv[++i]);
    }

    MatrixMultiplier mm(block_size);
    if (!mm.readMatrices(fileA, fileB)) return 1;

    double time_taken = mm.multiplyCUDA();
    cout << "Time: " << fixed << setprecision(4) << time_taken << "s (BS: " << block_size << ")" << endl;

    mm.writeResult(fileOut);
    return 0;
}