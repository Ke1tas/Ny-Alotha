import numpy as np
import time
import os
import subprocess
import matplotlib.pyplot as plt


def run_experiment(size, block_size, executable="./matrix_mul_cuda"):
    file_a, file_b, file_out = "a.txt", "b.txt", "out.txt"

    A = np.random.randint(-10, 10, (size, size))
    B = np.random.randint(-10, 10, (size, size))
    np.savetxt(file_a, A, fmt='%d')
    np.savetxt(file_b, B, fmt='%d')

    cmd = [executable, "-a", file_a, "-b", file_b, "-o", file_out, "-bs", str(block_size)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        output = result.stdout
        elapsed = float(output.split("Time: ")[1].split("s")[0])

        C_cpp = np.loadtxt(file_out)
        C_py = np.dot(A, B)
        if not np.array_equal(C_cpp, C_py):
            print(f"Validation failed for size {size}, BS {block_size}!")
            return None

        return elapsed
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        for f in [file_a, file_b, file_out]:
            if os.path.exists(f): os.remove(f)


def main():
    sizes = [256, 512, 1024, 1536, 2048]
    block_sizes = [4, 8, 16, 32]
    executable = "matrix_mul_cuda.exe"

    results = {bs: [] for bs in block_sizes}

    for bs in block_sizes:
        print(f"Testing Block Size: {bs}x{bs}")
        for size in sizes:
            t = run_experiment(size, bs, executable)
            print(f"  Size {size}x{size}: {t:.4f}s")
            results[bs].append(t)

    plt.figure(figsize=(10, 6))
    for bs in block_sizes:
        plt.plot(sizes, results[bs], marker='o', label=f'Block {bs}x{bs}')

    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('CUDA Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('cuda_performance.png')
    plt.show()


if __name__ == "__main__":
    main()