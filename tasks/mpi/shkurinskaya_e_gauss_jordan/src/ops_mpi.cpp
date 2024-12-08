#include "mpi/shkurinskaya_e_gauss_jordan/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <functional>
#include <random>
using namespace std::chrono;
#include <boost/serialization/serialization.hpp>
#include <string>
#include <thread>
#include <vector>

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  n = *reinterpret_cast<int*>(taskData->inputs[0]);
  matrix = std::vector<double>(reinterpret_cast<double*>(taskData->inputs[1]),
                               reinterpret_cast<double*>(taskData->inputs[1]) + n * (n + 1));
  solution = std::vector<double>(n, 0.0);
  return true;
}

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential::validation() {
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
      std::cout << "Warning: Zero diagonal element" << std::endl;
      return false;
    }
  }
  return true;
}

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential::run() {
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

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < n; ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = solution[i];
  }
  return true;
}

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    n = *reinterpret_cast<int*>(taskData->inputs[0]);
    int num_elements = n * (n + 1);
    solution = std::vector<double>();
    matrix = std::vector<double>(reinterpret_cast<double*>(taskData->inputs[1]),
                                 reinterpret_cast<double*>(taskData->inputs[1]) + num_elements);
    diag_elements.resize(n);
    for (int i = 0; i < n; ++i) {
      diag_elements[i] = (i * (n + 1) + i);
    }
  }

  return true;
}

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int numRows = taskData->inputs_count[0];
    int numCols = (taskData->inputs_count[0] > 0) ? (numRows + 1) : 0;
    if (numRows <= 0 || numCols <= 0) {
      std::cout << "Validation failed: invalid dimensions (rows or columns cannot be zero or negative)!" << std::endl;
      return false;
    }
    auto expectedSize = static_cast<size_t>(numRows * numCols);
    if (taskData->inputs_count[1] != expectedSize) {
      std::cout << "Validation failed: matrix size mismatch! Expected " << expectedSize << " elements, but found "
                << taskData->inputs_count[1] << " elements." << std::endl;
      return false;
    }
    auto* matrixData = reinterpret_cast<double*>(taskData->inputs[1]);
    for (int i = 0; i < numRows; ++i) {
      double diagElement = matrixData[i * (numCols) + i];
      if (std::abs(diagElement) < 1e-9) {
        std::cout << "Validation failed: Zero or near-zero diagonal element at row " << i << ", value: " << diagElement
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);
  for (int k = 0; k < n; ++k) {
    if (world.rank() == 0) {
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
      header = std::vector<double>(matrix.begin() + (k * (n + 1)), matrix.begin() + (k * (n + 1)) + n + 1);
      int offset = (n + 1) * (k + 1);
      int remainderSize = matrix.size() - offset;
      int elements_per_process = ((remainderSize / (n + 1)) / world.size()) * (n + 1);
      int remainder = ((remainderSize / (n + 1)) % world.size()) * (n + 1);
      sendCounts = std::vector<int>(world.size(), elements_per_process);
      for (int i = 0; i < remainder / (n + 1); i++) {
        sendCounts[i] += (n + 1);
      }
      displacements = std::vector<int>(world.size(), offset);
      for (int i = 1; i < world.size(); ++i) {
        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
      }
    }
    boost::mpi::broadcast(world, header, 0);
    boost::mpi::broadcast(world, sendCounts, 0);
    boost::mpi::broadcast(world, displacements, 0);

    localMatrix.resize(sendCounts[world.rank()]);
    boost::mpi::scatterv(world, matrix, sendCounts, displacements, localMatrix.data(), sendCounts[world.rank()], 0);
    for (size_t i = 0; i < (localMatrix.size() / (n + 1)); ++i) {
      double factor = localMatrix[i * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        localMatrix[i * (n + 1) + j] -= header[j] * factor;
      }
    }
    boost::mpi::gatherv(world, localMatrix, matrix.data(), sendCounts, displacements, 0);
  }
  for (int k = n - 1; k >= 0; --k) {
    if (world.rank() == 0) {
      header = std::vector<double>(matrix.begin() + (k * (n + 1)), matrix.begin() + (k * (n + 1)) + n + 1);

      int offset = (n + 1) * (k);
      int remainderSize = offset;
      int elements_per_process = ((remainderSize / (n + 1)) / world.size()) * (n + 1);
      int remainder = ((remainderSize / (n + 1)) % world.size()) * (n + 1);

      sendCounts = std::vector<int>(world.size(), elements_per_process);
      for (int i = 0; i < remainder / (n + 1); i++) {
        sendCounts[i] += (n + 1);
      }

      displacements = std::vector<int>(world.size(), 0);
      for (int i = 1; i < world.size(); ++i) {
        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
      }
    }
    boost::mpi::broadcast(world, header, 0);
    boost::mpi::broadcast(world, sendCounts, 0);
    boost::mpi::broadcast(world, displacements, 0);

    localMatrix.resize(sendCounts[world.rank()]);
    boost::mpi::scatterv(world, matrix, sendCounts, displacements, localMatrix.data(), sendCounts[world.rank()], 0);
    for (size_t i = 0; i < (localMatrix.size() / (n + 1)); ++i) {
      double factor = localMatrix[i * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        localMatrix[i * (n + 1) + j] -= header[j] * factor;
      }
    }
    boost::mpi::gatherv(world, localMatrix, matrix.data(), sendCounts, displacements, 0);
  }
  return true;
}

bool shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = matrix[i * (n + 1) + n];
    }
  }
  return true;
}
