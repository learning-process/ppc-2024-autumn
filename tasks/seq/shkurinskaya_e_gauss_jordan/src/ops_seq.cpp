#include "seq/shkurinskaya_e_gauss_jordan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace shkurinskaya_e_gauss_jordan_seq;

bool shkurinskaya_e_gauss_jordan_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  n = *reinterpret_cast<int*>(taskData->inputs[0]);
  matrix = std::vector<double>(reinterpret_cast<double*>(taskData->inputs[1]),
                               reinterpret_cast<double*>(taskData->inputs[1]) + n * (n + 1));
  solution = std::vector<double>(n, 0.0);
  return true;
}

bool shkurinskaya_e_gauss_jordan_seq::TestTaskSequential::validation() {
  internal_order_test();

  int numRows = taskData->inputs_count[0];
  int numCols = (taskData->inputs_count[0] > 0) ? (numRows + 1) : 0;
  if (numRows <= 0 || numCols <= 0) {
    std::cout << "Validation failed: invalid dimensions (rows or columns cannot be zero or negative)!" << std::endl;
    return false;
  }
  auto expectedSize = static_cast<size_t>(numRows * numCols);
  if (taskData->inputs_count[1] != expectedSize) {
    std::cout << "Validation failed: matrix size mismatch!" << std::endl;
    return false;
  }
  auto* matrixData = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < numRows; ++i) {
    auto value = matrixData[i * numCols + i];
    if (value == 0.0) {
      std::cout << "Warning: Zero diagonal element at index " << i << std::endl;
      return false;
    }
  }
  return true;
}

bool shkurinskaya_e_gauss_jordan_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int k = 0; k < n; ++k) {
    int max_row = k;
    for (int i = k + 1; i < n; ++i) {
      if (std::abs(matrix[i * (n + 1) + k]) > std::abs(matrix[max_row * (n + 1) + k])) {
        max_row = i;
      }
    }
    if (max_row != k) {
      for (int j = k; j <= n; ++j) {
        std::swap(matrix[k * (n + 1) + j], matrix[max_row * (n + 1) + j]);
      }
    }
    double diag = matrix[k * (n + 1) + k];
    for (int j = k; j <= n; ++j) {
      matrix[k * (n + 1) + j] /= diag;
    }
    for (int i = k + 1; i < n; ++i) {
      double factor = matrix[i * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        matrix[i * (n + 1) + j] -= matrix[k * (n + 1) + j] * factor;
      }
    }
  }
  for (int k = n - 1; k >= 0; --k) {
    for (int i = k - 1; i >= 0; --i) {
      double factor = matrix[i * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        matrix[i * (n + 1) + j] -= matrix[k * (n + 1) + j] * factor;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    solution[i] = matrix[i * (n + 1) + n];
  }

  return true;
}

bool shkurinskaya_e_gauss_jordan_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  for (int i = 0; i < n; ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = solution[i];
  }
  return true;
}
