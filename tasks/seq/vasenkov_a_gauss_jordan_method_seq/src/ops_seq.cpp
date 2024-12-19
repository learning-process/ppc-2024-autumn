#include "seq/vasenkov_a_gauss_jordan_method_seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

std::vector<double> vasenkov_a_gauss_jordan_method_seq::processMatrix(int numRows, int numCols,
                                                                      const std::vector<double>& inputMatrix) {
  if (numRows <= 0 || numCols < 0 || numCols >= numRows) {
    throw std::invalid_argument("Invalid matrix dimensions.");
  }

  std::vector<double> resultVector(numRows * (numRows - numCols + 1));

  for (int i = 0; i < (numRows - numCols + 1); i++) {
    resultVector[i] = inputMatrix[(numRows + 1) * numCols + numCols + i];
  }

  for (int row = 0; row < numCols; row++) {
    for (int col = 0; col < (numRows - numCols + 1); col++) {
      resultVector[(numRows - numCols + 1) * (row + 1) + col] = inputMatrix[row * (numRows + 1) + numCols + col];
    }
  }

  for (int row = numCols + 1; row < numRows; row++) {
    for (int col = 0; col < (numRows - numCols + 1); col++) {
      resultVector[(numRows - numCols + 1) * row + col] = inputMatrix[row * (numRows + 1) + numCols + col];
    }
  }

  return resultVector;
}

void vasenkov_a_gauss_jordan_method_seq::updateMatrix(int numRows, int numCols, std::vector<double>& matrix,
                                                      const std::vector<double>& iterationResults) {
  if (numRows <= 0 || numCols < 0 || numCols >= numRows) {
    throw std::invalid_argument("Invalid matrix dimensions.");
  }

  for (int row = 0; row < numCols; row++) {
    for (int col = 0; col < (numRows - numCols); col++) {
      matrix[row * (numRows + 1) + numCols + 1 + col] = iterationResults[row * (numRows - numCols) + col];
    }
  }

  for (int row = numCols + 1; row < numRows; row++) {
    for (int col = 0; col < (numRows - numCols); col++) {
      matrix[row * (numRows + 1) + numCols + 1 + col] = iterationResults[(row - 1) * (numRows - numCols) + col];
    }
  }

  double diagonalElement = matrix[numCols * (numRows + 1) + numCols];
  if (diagonalElement == 0.0) {
    throw std::runtime_error("Division by zero during normalization.");
  }

  for (int i = numCols + 1; i < numRows + 1; i++) {
    matrix[numCols * (numRows + 1) + i] /= diagonalElement;
  }

  for (int i = 0; i < numRows; i++) {
    matrix[i * (numRows + 1) + numCols] = 0;
  }

  matrix[numCols * (numRows + 1) + numCols] = 1;
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
