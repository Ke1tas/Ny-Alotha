import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt


def generate_matrices(size, min_val=-10, max_val=10):
    """Генерация случайных квадратных матриц"""
    np.random.seed(42)  # Для воспроизводимости
    A = np.random.randint(min_val, max_val, (size, size))
    B = np.random.randint(min_val, max_val, (size, size))
    return A, B


def save_matrix(filename, matrix):
    """Сохранение матрицы в файл"""
    np.savetxt(filename, matrix, fmt='%.0f', delimiter=' ')


def load_matrix(filename):
    """Загрузка матрицы из файла"""
    return np.loadtxt(filename)


def verify_result(c_matrix_file, expected_C):
    """Верификация результата"""
    try:
        C_cpp = load_matrix(c_matrix_file)

        if C_cpp.shape != expected_C.shape:
            print(f"❌ Ошибка: Размеры матриц не совпадают!")
            print(f"   Ожидалось: {expected_C.shape}, Получено: {C_cpp.shape}")
            return False

        max_diff = np.max(np.abs(C_cpp - expected_C))
        mean_diff = np.mean(np.abs(C_cpp - expected_C))

        tolerance = 1e-5
        is_correct = np.allclose(C_cpp, expected_C, rtol=tolerance, atol=tolerance)

        print(f"\nРЕЗУЛЬТАТЫ ВЕРИФИКАЦИИ:")
        print(f"Максимальная разница: {max_diff:.6e}")
        print(f"Средняя разница: {mean_diff:.6e}")
        print(f"Допуск: {tolerance}")

        if is_correct:
            print("  ✅ Верификация пройдена!")
        else:
            print("  ❌ Верификация не пройдена!")

        return is_correct

    except Exception as e:
        print(f"  ❌ Ошибка при верификации: {e}")
        return False


def plot_results(sizes, times):
    """Построение графика времени выполнения"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Размер матрицы (N x N)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени умножения матриц от размера', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Добавляем значения на график
    for i, (size, t) in enumerate(zip(sizes, times)):
        plt.annotate(f'{t:.3f}с', (size, t), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('matrix_multiplication_time.png', dpi=150)
    plt.show()


def main():
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДАННЫХ И ТЕСТИРОВАНИЕ УМНОЖЕНИЯ МАТРИЦ")
    print("=" * 60)

    matrix_sizes = [200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
    files = {
        'A': 'matrix_a.txt',
        'B': 'matrix_b.txt',
        'C': 'matrix_c.txt'
    }

    times = []
    verified_sizes = []

    for i, size in enumerate(matrix_sizes, 1):
        print(f"\n{'=' * 60}")
        print(f"ТЕСТ {i}/{len(matrix_sizes)}: Размер матрицы {size}x{size}")
        print(f"{'=' * 60}")

        # Генерация матриц
        print(f"  Генерация матриц...")
        A, B = generate_matrices(size)

        # Сохранение матриц
        print(f"  Сохранение матриц в файлы...")
        save_matrix(files['A'], A)
        save_matrix(files['B'], B)


        expected_C = np.matmul(A, B)


        cmd = f"matrix_mul.exe -a {files['A']} -b {files['B']} -o {files['C']}"

        start_time = time.time()
        result = os.system(cmd)
        cpp_time = time.time() - start_time

        if result != 0:
            print(f"  ❌ Ошибка при выполнении C++ программы!")
            continue

        times.append(cpp_time)
        verified_sizes.append(size)

        print(f"  ⏱️  Время выполнения C++: {cpp_time:.3f} сек")

        verify_result(files['C'], expected_C)

        for file in files.values():
            if os.path.exists(file):
                os.remove(file)

    # Вывод итоговой таблицы
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"{'Размер':<10} {'Время (сек)':<15} {'Элементов':<15}")
    print("-" * 40)
    for size, t in zip(verified_sizes, times):
        print(f"{size:<10} {t:<15.3f} {size * size:<15,}")

    # Построение графика
    if times:
        print("\n  Построение графика...")
        plot_results(verified_sizes, times)
        print("  ✓ График сохранен как 'matrix_multiplication_time.png'")
    else:
        print("\n  ❌ Нет данных для построения графика")


if __name__ == "__main__":
    main()