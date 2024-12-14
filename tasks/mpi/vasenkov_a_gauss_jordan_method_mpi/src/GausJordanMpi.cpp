#include "mpi/vasenkov_a_gauss_jordan_method_mpi/include/GausJordanMpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define EPSILON 1e-9

std::vector<double> vasenkov_a_gauss_jordan_method_mpi::processMatrix(int numRows, int numCols,
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

void vasenkov_a_gauss_jordan_method_mpi::updateMatrix(int numRows, int numCols, std::vector<double>& matrix,
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

void vasenkov_a_gauss_jordan_method_mpi::calcSizesDispls(int n, int k, int world_size, std::vector<int>& sizes,
                                                         std::vector<int>& displs) {
  int r = n - 1;
  int c = n - k;
  sizes.resize(world_size, 0);
  displs.resize(world_size, 0);

  if (world_size > r) {
    for (int i = 0; i < r; ++i) {
      sizes[i] = c;
      displs[i] = i * c;
    }
  } else {
    int a = r / world_size;
    int b = r % world_size;

    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * c;
      } else {
        sizes[i] = a * c;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

std::vector<std::pair<int, int>> vasenkov_a_gauss_jordan_method_mpi::getIndicies(int rows, int cols) {
  std::vector<std::pair<int, int>> indicies;
  indicies.reserve(rows * cols);

  for (int i = 1; i < rows; ++i) {
    for (int j = 1; j < cols; ++j) {
      indicies.emplace_back(i, j);
    }
  }
  return indicies;
}

bool vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel::validation() {
  internal_order_test();
  if (world.rank() != 0) {
    return true;
  }
  int n_val = *reinterpret_cast<int*>(taskData->inputs[1]);
  int matrix_size = taskData->inputs_count[0];
  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);

  if (n_val * (n_val + 1) == matrix_size) {
    std::vector<double> temp_matrix(matrix_size);
    temp_matrix.assign(matrix_data, matrix_data + matrix_size);
    return true;
  }
  return false;
}

bool vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
    int matrix_size = taskData->inputs_count[0];

    n = *reinterpret_cast<int*>(taskData->inputs[1]);

    matrix.assign(matrix_data, matrix_data + matrix_size);
  }

  return true;
}

bool vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, n, 0);

  for (int k = 0; k < n; k++) {
    if (world.rank() == 0) {
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
        if (change == n) {
          solve = false;
        }
      }

      if (solve) {
        iter_matrix = vasenkov_a_gauss_jordan_method_mpi::processMatrix(n, k, matrix);

        vasenkov_a_gauss_jordan_method_mpi::calcSizesDispls(n, k, world.size(), sizes, displs);
        indicies = vasenkov_a_gauss_jordan_method_mpi::getIndicies(n, n - k + 1);

        iter_result.resize((n - 1) * (n - k));
      }
    }
    boost::mpi::broadcast(world, solve, 0);
    if (!solve) return false;
    boost::mpi::broadcast(world, sizes, 0);
    boost::mpi::broadcast(world, iter_matrix, 0);

    int local_size = sizes[world.rank()];
    std::vector<std::pair<int, int>> local_indicies(local_size);
    if (world.rank() == 0) {
      boost::mpi::scatterv(world, indicies.data(), sizes, displs, local_indicies.data(), local_size, 0);
    } else {
      boost::mpi::scatterv(world, local_indicies.data(), local_size, 0);
    }

    std::vector<double> local_result;
    local_result.reserve(local_size);
    for (int ind = 0; ind < local_size; ind++) {
      auto [i, j] = local_indicies[ind];
      double rel = iter_matrix[0];
      double nel = iter_matrix[i * (n - k + 1) + j];
      double a = iter_matrix[j];
      double b = iter_matrix[i * (n - k + 1)];
      double res = nel - (a * b) / rel;
      local_result[ind] = res;
    }

    if (world.rank() == 0) {
      boost::mpi::gatherv(world, local_result.data(), local_size, iter_result.data(), sizes, displs, 0);
    } else {
      boost::mpi::gatherv(world, local_result.data(), local_size, 0);
    }

    if (world.rank() == 0) {
      vasenkov_a_gauss_jordan_method_mpi::updateMatrix(n, k, matrix, iter_result);
    }
  }

  return true;
}

bool vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel::post_processing() {
  internal_order_test();
  if (!solve) {
    return false;
  }
  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(matrix.begin(), matrix.end(), output_data);
  }

  return true;
}