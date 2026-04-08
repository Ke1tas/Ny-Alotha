import numpy as np
import time
import sys
import os


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
            print("✅ Верификация пройдена: результаты совпадают!")
        else:
            print("❌ Верификация не пройдена: результаты отличаются!")

            # Вывод примеров различий
            diff_indices = np.where(np.abs(C_cpp - expected_C) > tolerance)
            print(f"\nНайдено {len(diff_indices[0])} значительных различий:")
            for i in range(min(5, len(diff_indices[0]))):
                idx = diff_indices[0][i], diff_indices[1][i]
                print(
                    f"  Позиция {idx}: C++={C_cpp[idx]:.6f}, Python={expected_C[idx]:.6f}, разница={abs(C_cpp[idx] - expected_C[idx]):.6f}")

        return is_correct

    except Exception as e:
        print(f"❌ Ошибка при верификации: {e}")
        return False


def main():
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДАННЫХ И ВЕРИФИКАЦИЯ УМНОЖЕНИЯ МАТРИЦ")
    print("=" * 60)

    # Параметры
    matrix_size = 200  # Размер матрицы
    files = {
        'A': 'matrix_a.txt',
        'B': 'matrix_b.txt',
        'C': 'matrix_c.txt'
    }

    print(f"\n1. Генерация матриц размером {matrix_size}x{matrix_size}...")
    A, B = generate_matrices(matrix_size)

    print("\n2. Сохранение матриц в файлы...")
    save_matrix(files['A'], A)
    save_matrix(files['B'], B)
    print(f"   ✓ Матрица A сохранена в {files['A']}")
    print(f"   ✓ Матрица B сохранена в {files['B']}")

    expected_C = np.matmul(A, B)

    print("\n4. Запуск C++ программы...")
    print("   Выполняется умножение матриц...")

    cmd = f"matrix_mul.exe -a {files['A']} -b {files['B']} -o {files['C']}"

    start_time = time.time()
    result = os.system(cmd)
    cpp_time = time.time() - start_time

    if result != 0:
        print("❌ Ошибка при выполнении C++ программы!")
        return

    print(f"\n5. Верификация результата...")

    # Проверка существования файла результата
    if not os.path.exists(files['C']):
        print(f"❌ Файл результата {files['C']} не найден!")
        return

    # Верификация
    verification_result = verify_result(files['C'], expected_C)

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    print(f"Размер матриц: {matrix_size}x{matrix_size}")
    print(f"\nВремя выполнения:")
    print(f"  C++ программа:   {cpp_time:.3f} сек")


if __name__ == "__main__":
    main()