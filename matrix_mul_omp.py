import numpy as np
import time
import os
import subprocess
import matplotlib.pyplot as plt


def generate_matrices(size, min_val=-10, max_val=10):
    np.random.seed(228322)
    A = np.random.randint(min_val, max_val, (size, size))
    B = np.random.randint(min_val, max_val, (size, size))
    return A, B


def save_matrix(filename, matrix):
    np.savetxt(filename, matrix, fmt='%.0f', delimiter=' ')


def load_matrix(filename):
    return np.loadtxt(filename)


def verify_result(c_matrix_file, expected_C):
    try:
        C_cpp = load_matrix(c_matrix_file)
        if C_cpp.shape != expected_C.shape:
            return False
        is_correct = np.allclose(C_cpp, expected_C, rtol=1e-5, atol=1e-5)
        return is_correct
    except Exception as e:
        return False


def run_test(size, num_threads, executable="matrix_mul_omp.exe"):
    files = {
        'A': f'matrix_a_{size}.txt',
        'B': f'matrix_b_{size}.txt',
        'C': f'matrix_c_{size}_t{num_threads}.txt'
    }

    try:
        A, B = generate_matrices(size)
        save_matrix(files['A'], A)
        save_matrix(files['B'], B)

        expected_C = np.matmul(A, B)

        cmd = [executable, "-a", files['A'], "-b", files['B'], "-o", files['C'], "-t", str(num_threads)]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            return None

        is_valid = verify_result(files['C'], expected_C)

        for file in files.values():
            if os.path.exists(file):
                os.remove(file)

        if is_valid:
            print(f'size: {size}, threads {num_threads} is valid \nelapsed time {elapsed_time}')
            return elapsed_time
        return None

    except Exception as e:
        return None


def plot_results(sizes, times_dict):
    plt.figure(figsize=(10, 6))

    for threads, times in times_dict.items():
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=f'{threads} threads')

    plt.xlabel('Matrix Size (N x N)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Matrix Multiplication Performance with OpenMP', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('matrix_multiplication_time.png', dpi=150)
    plt.show()


def main():
    matrix_sizes = [200, 400, 800, 1200, 1600, 2000]
    threads_to_test = [1, 2, 4, 8]
    executable = "matrix_mul_omp.exe"

    if not os.path.exists(executable):
        print(f"Error: {executable} not found!")
        return

    results = {threads: [] for threads in threads_to_test}

    for threads in threads_to_test:
        for size in matrix_sizes:
            time_val = run_test(size, threads, executable)
            if time_val:
                results[threads].append(time_val)
            else:
                results[threads].append(None)

    valid_results = {threads: [] for threads in threads_to_test}
    valid_sizes = []

    for i, size in enumerate(matrix_sizes):
        all_valid = all(results[t][i] is not None for t in threads_to_test)
        if all_valid:
            valid_sizes.append(size)
            for threads in threads_to_test:
                valid_results[threads].append(results[threads][i])

    if valid_sizes:
        plot_results(valid_sizes, valid_results)


if __name__ == "__main__":
    main()