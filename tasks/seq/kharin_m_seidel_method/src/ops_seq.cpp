// ops_seq.cpp
#include "seq/kharin_m_seidel_method/include/ops_seq.hpp"

#include <cmath>

bool kharin_m_seidel_method::GaussSeidelSequential::pre_processing() {
  internal_order_test();

  // Чтение eps из taskData
  eps = *(reinterpret_cast<double*>(taskData->inputs[1]));

  a.resize(n * n);
  b.resize(n);
  x.resize(n, 1.0);  // Инициализация x значением 1.0
  p.resize(n);

  // Чтение матрицы A из taskData->inputs[2]
  auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);
  std::copy(a_data, a_data + n * n, a.begin());

  // Чтение вектора b из taskData->inputs[3]
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[3]);
  std::copy(b_data, b_data + n, b.begin());

  return true;
}

bool kharin_m_seidel_method::GaussSeidelSequential::validation() {
  internal_order_test();
  bool is_valid = true;
  n = *(reinterpret_cast<int*>(taskData->inputs[0]));
  // Проверка размеров входных данных
  if (taskData->inputs_count[0] != static_cast<size_t>(1) || taskData->inputs_count[1] != static_cast<size_t>(1) ||
      taskData->inputs_count[2] != static_cast<size_t>(n * n) || taskData->inputs_count[3] != static_cast<size_t>(n) ||
      taskData->outputs_count[0] != static_cast<size_t>(n)) {
    is_valid = false;
  }

  if (is_valid) {
    // Чтение матрицы A из taskData->inputs[2]
    auto* a_data = reinterpret_cast<double*>(taskData->inputs[2]);

    // Проверка условия сходимости и единственности решения
    for (int i = 0; i < n; ++i) {
      double diag = std::abs(a_data[i * n + i]);
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += std::abs(a_data[i * n + j]);
        }
      }
      if (diag <= sum) {
        is_valid = false;
        break;
      }
    }
    if (is_valid) {
      // Метод Гаусса для определения ранга
      int rank = 0;
      std::vector<bool> row_used(n, false);
      for (int j = 0; j < n; ++j) {
        int pivot_row = -1;
        for (int i = 0; i < n; ++i) {
          if (!row_used[i] && std::abs(a_data[i * n + j]) > 1e-10) {
            pivot_row = i;
            break;
          }
        }
        if (pivot_row == -1) continue;
        row_used[pivot_row] = true;
        rank++;
        // Вычитание строк для приведения к треугольному виду
        for (int i = 0; i < n; ++i) {
          if (i != pivot_row && std::abs(a_data[i * n + j]) > 1e-10) {
            double factor = a_data[i * n + j] / a_data[pivot_row * n + j];
            for (int k = j; k < n; ++k) {
              a_data[i * n + k] -= factor * a_data[pivot_row * n + k];
            }
          }
        }
      }
      // Проверяем, полный ли ранг
      is_valid = (rank == n);
    }
  }
  return is_valid;
}

bool kharin_m_seidel_method::GaussSeidelSequential::run() {
  internal_order_test();

  bool converged = false;
  int m = 0;

  while (!converged && m < max_iterations) {
    // Копирование x в p
    p = x;

    // Обновление x
    for (int i = 0; i < n; i++) {
      double var = 0.0;
      for (int j = 0; j < n; j++) {
        if (j != i) {
          var += a[i * n + j] * x[j];
        }
      }
      x[i] = (b[i] - var) / a[i * n + i];
    }

    // Проверка сходимости
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
      norm += (x[i] - p[i]) * (x[i] - p[i]);
    }
    converged = (sqrt(norm) < eps);
    m++;
  }

  return true;
}

bool kharin_m_seidel_method::GaussSeidelSequential::post_processing() {
  internal_order_test();

  // Запись результатов в taskData->outputs[0]
  auto* x_output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x.begin(), x.end(), x_output);
  return true;
}