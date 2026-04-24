#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <mpi.h>
#include <algorithm>

using namespace std;

typedef long long int big_int;

// TILE_SIZE 64 is generally optimal for most modern L2/L3 caches
const int TILE_SIZE = 64;

class MatrixMultiplierMPI {
private:
    int rank, world_size;
    int N; // Matrix size

public:
    MatrixMultiplierMPI(int r, int s) : rank(r), world_size(s), N(0) {}

    // Process 0 reads the file to determine dimensions
    bool getMatrixSize(const string& filename, int& size) {
        ifstream file(filename);
        if (!file.is_open()) return false;
        string line;
        int count = 0;
        if (getline(file, line)) {
            stringstream ss(line);
            big_int val;
            while (ss >> val) count++;
        }
        size = count;
        return size > 0;
    }

    // Process 0 reads the matrix into a flat contiguous vector
    void readMatrix(const string& filename, vector<big_int>& matrix, int size) {
        ifstream file(filename);
        matrix.resize(size * size);
        for (int i = 0; i < size * size; i++) {
            file >> matrix[i];
        }
    }

    void run(const string& fileA, const string& fileB, const string& fileOut) {
        vector<big_int> A_full, B_all, C_full;

        // 1. Metadata synchronization
        if (rank == 0) {
            if (!getMatrixSize(fileA, N)) {
                cerr << "Error reading matrix A" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 2. Calculate distribution (handling non-divisible N)
        vector<int> sendcounts(world_size);
        vector<int> displs(world_size);
        int sum = 0;
        for (int i = 0; i < world_size; i++) {
            int rows = (N / world_size) + (i < (N % world_size) ? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = sum;
            sum += sendcounts[i];
        }

        int local_rows = sendcounts[rank] / N;
        vector<big_int> A_local(sendcounts[rank]);
        vector<big_int> C_local(sendcounts[rank], 0);
        B_all.resize(N * N);

        // 3. Load and Distribute Data
        if (rank == 0) {
            readMatrix(fileA, A_full, N);
            readMatrix(fileB, B_all, N);
        }

        // Send B to all, and slices of A to each process
        MPI_Bcast(B_all.data(), N * N, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_LONG_LONG_INT,
            A_local.data(), sendcounts[rank], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        // 4. TILED Computation
        double start_time = MPI_Wtime();

        for (int i_tile = 0; i_tile < local_rows; i_tile += TILE_SIZE) {
            for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
                for (int j_tile = 0; j_tile < N; j_tile += TILE_SIZE) {

                    // Multiply within the block/tile
                    for (int i = i_tile; i < min(i_tile + TILE_SIZE, local_rows); ++i) {
                        for (int k = k_tile; k < min(k_tile + TILE_SIZE, N); ++k) {
                            big_int a_val = A_local[i * N + k];
                            // Inner loop is j-loop for best cache stride
                            for (int j = j_tile; j < min(j_tile + TILE_SIZE, N); ++j) {
                                C_local[i * N + j] += a_val * B_all[k * N + j];
                            }
                        }
                    }

                }
            }
        }

        double end_time = MPI_Wtime();
        double local_elapsed = end_time - start_time;
        double max_elapsed;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                // 5. Gather and Save
        if (rank == 0) C_full.resize(N * N);
        MPI_Gatherv(C_local.data(), sendcounts[rank], MPI_LONG_LONG_INT,
            C_full.data(), sendcounts.data(), displs.data(), MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Calculation finished in " << fixed << setprecision(4) << max_elapsed << " seconds." << endl;

            // Save Matrix
            ofstream file(fileOut);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    file << C_full[i * N + j] << (j == N - 1 ? "" : " ");
                }
                file << "\n";
            }
            file.close();

            // NEW: Save timing information to a separate file
            ofstream tfile(fileOut + ".time");
            tfile << fixed << setprecision(6) << max_elapsed;
            tfile.close();
        }
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    string fileA = "matrix_a.txt";
    string fileB = "matrix_b.txt";
    string fileOut = "matrix_c.txt";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-a" && i + 1 < argc) fileA = argv[++i];
        else if (arg == "-b" && i + 1 < argc) fileB = argv[++i];
        else if (arg == "-o" && i + 1 < argc) fileOut = argv[++i];
    }

    if (rank == 0) {
        cout << "Running with " << world_size << " MPI processes." << endl;
    }

    MatrixMultiplierMPI multiplier(rank, world_size);
    multiplier.run(fileA, fileB, fileOut);

    MPI_Finalize();
    return 0;
}