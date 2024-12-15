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

std::vector<double> vasenkov_a_gauss_jordan_method_mpi::processMatrix(int rows, int cols,
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

void vasenkov_a_gauss_jordan_method_mpi::updateMatrix(int rows, int cols, std::vector<double>& mat,
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
std::vector<std::pair<int, int>> getIndex(int rowCount, int colCount) {
  std::vector<std::pair<int, int>> indexPairs;
  indexPairs.reserve(rowCount * colCount);

  for (int i = 1; i < rowCount; ++i) {
    for (int j = 1; j < colCount; ++j) {
      indexPairs.emplace_back(i, j);
    }
  }

  return indexPairs;
}

void processMatrixForPivot(int pivot, std::vector<double>& matrix, int n, bool& solve) {
  if (matrix[pivot * (n + 1) + pivot] == 0) {
    int rowToSwap;

    for (rowToSwap = pivot + 1; rowToSwap < n; ++rowToSwap) {
      if (matrix[rowToSwap * (n + 1) + pivot] != 0) {
        for (int column = 0; column < (n + 1); ++column) {
          std::swap(matrix[pivot * (n + 1) + column], matrix[rowToSwap * (n + 1) + column]);
        }
        break;
      }
    }

    if (rowToSwap == n) {
      solve = false;
    }
  }
}

bool vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, n, 0);

  for (int pivot = 0; pivot < n; ++pivot) {
    if (world.rank() == 0) {
      processMatrixForPivot(pivot, matrix, n, solve);

      if (solve) {
        iteration_matrix = vasenkov_a_gauss_jordan_method_mpi::processMatrix(n, pivot, matrix);

        int remainingRows = n - 1;
        int remainingCols = n - pivot;

        sizes.resize(world.size(), 0);
        displaces.resize(world.size(), 0);

        if (world.size() > remainingRows) {
          for (int i = 0; i < remainingRows; ++i) {
            sizes[i] = remainingCols;
            displaces[i] = i * remainingCols;
          }
        } else {
          int baseSize = remainingRows / world.size();
          int extraRows = remainingRows % world.size();

          int currentOffset = 0;
          for (int i = 0; i < world.size(); ++i) {
            if (extraRows-- > 0) {
              sizes[i] = (baseSize + 1) * remainingCols;
            } else {
              sizes[i] = baseSize * remainingCols;
            }
            displaces[i] = currentOffset;
            currentOffset += sizes[i];
          }
        }

        current_index = getIndex(n, n - pivot + 1);
        result.resize((n - 1) * (n - pivot));
      }
    }

    boost::mpi::broadcast(world, solve, 0);

    if (!solve) return false;

    boost::mpi::broadcast(world, sizes, 0);
    boost::mpi::broadcast(world, iteration_matrix, 0);

    int localCount = sizes[world.rank()];
    std::vector<std::pair<int, int>> localIndices(localCount);

    if (world.rank() == 0) {
      boost::mpi::scatterv(world, current_index.data(), sizes, displaces, localIndices.data(), localCount, 0);
    } else {
      boost::mpi::scatterv(world, localIndices.data(), localCount, 0);
    }

    std::vector<double> localResults(localCount);

    for (int index = 0; index < localCount; ++index) {
      auto [rowIndex, colIndex] = localIndices[index];

      double referenceValue = iteration_matrix[0];
      double currentValue = iteration_matrix[rowIndex * (n - pivot + 1) + colIndex];

      double factor1 = iteration_matrix[colIndex];
      double factor2 = iteration_matrix[rowIndex * (n - pivot + 1)];

      localResults[index] = currentValue - (factor1 * factor2) / referenceValue;
    }

    if (world.rank() == 0) {
      boost::mpi::gatherv(world, localResults.data(), localCount, result.data(), sizes, displaces, 0);
    } else {
      boost::mpi::gatherv(world, localResults.data(), localCount, 0);
    }

    if (world.rank() == 0) {
      vasenkov_a_gauss_jordan_method_mpi::updateMatrix(n, pivot, matrix, result);
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
