#include "mpi/vasenkov_a_gauss_jordan_method_mpi/include/GausJordanSeq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

std::vector<double> vasenkov_a_gauss_jordan_method_seq::processMatrix(int rows, int cols,
                                                                      const std::vector<double>& srcMatrix) {
  if (rows <= 0 || cols < 0 || cols >= rows) {
    throw std::invalid_argument("Неверные размеры матрицы.");
  }

  std::vector<double> outputVector(rows * (rows - cols + 1));

  for (int i = 0; i < (rows - cols + 1); i++) {
    outputVector[i] = srcMatrix[(rows + 1) * cols + cols + i];
  }

  for (int r = 0; r < cols; r++) {
    for (int c = 0; c < (rows - cols + 1); c++) {
      outputVector[(rows - cols + 1) * (r + 1) + c] = srcMatrix[r * (rows + 1) + cols + c];
    }
  }

  for (int r = cols + 1; r < rows; r++) {
    for (int c = 0; c < (rows - cols + 1); c++) {
      outputVector[(rows - cols + 1) * r + c] = srcMatrix[r * (rows + 1) + cols + c];
    }
  }

  return outputVector;
}

void vasenkov_a_gauss_jordan_method_seq::updateMatrix(int rows, int cols, std::vector<double>& mat,
                                                      const std::vector<double>& results) {
  if (rows <= 0 || cols < 0 || cols >= rows) {
    throw std::invalid_argument("Неверные размеры матрицы.");
  }

  for (int r = 0; r < cols; r++) {
    for (int c = 0; c < (rows - cols); c++) {
      mat[r * (rows + 1) + cols + 1 + c] = results[r * (rows - cols) + c];
    }
  }

  for (int r = cols + 1; r < rows; r++) {
    for (int c = 0; c < (rows - cols); c++) {
      mat[r * (rows + 1) + cols + 1 + c] = results[(r - 1) * (rows - cols) + c];
    }
  }

  double diagElem = mat[cols * (rows + 1) + cols];

  if (diagElem == 0.0) {
    throw std::runtime_error("Деление на ноль во время нормализации.");
  }

  for (int i = cols + 1; i < rows + 1; i++) {
    mat[cols * (rows + 1) + i] /= diagElem;
  }

  for (int i = 0; i < rows; i++) {
    mat[i * (rows + 1) + cols] = 0;
  }

  mat[cols * (rows + 1) + cols] = 1;
}

bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::validation() {
  internal_order_test();

  int n_val = *reinterpret_cast<int*>(taskData->inputs[1]);
  int matrix_size = taskData->inputs_count[0];

  return n_val * (n_val + 1) == matrix_size;
}

bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::pre_processing() {
  internal_order_test();

  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  n = *reinterpret_cast<int*>(taskData->inputs[1]);

  if (n <= 0 || matrix_size != n * (n + 1)) {
    throw std::invalid_argument("Invalid input data.");
  }

  matrix.assign(matrix_data, matrix_data + matrix_size);

  return true;
}

bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::run() {
  internal_order_test();

  for (int k = 0; k < n; k++) {
    if (matrix[k * (n + 1) + k] == 0) {
      int change;
      for (change = k + 1; change < n; change++) {
        if (matrix[change * (n + 1) + k] != 0) {
          for (int col = 0; col < (n + 1); col++) {
            std::swap(matrix[k * (n + 1) + col], matrix[change * (n + 1) + col]);
          }
          break;
        }
      }
      if (change == n) return false;
    }

    auto iterMatrix = vasenkov_a_gauss_jordan_method_seq::processMatrix(n, k, matrix);

    std::vector<double> iterResult((n - 1) * (n - k));

    int ind = 0;
    for (int i = 1; i < n; ++i) {
      for (int j = 1; j < n - k + 1; ++j) {
        double rel = iterMatrix[0];
        double nel = iterMatrix[i * (n - k + 1) + j];
        double a = iterMatrix[j];
        double b = iterMatrix[i * (n - k + 1)];
        iterResult[ind++] = nel - a * b / rel;
      }
    }

    vasenkov_a_gauss_jordan_method_seq::updateMatrix(n, k, matrix, iterResult);
  }

  return true;
}

bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);

  std::copy(matrix.begin(), matrix.end(), output_data);

  return true;
}
