// Copyright 2023 Nesterov Alexander
#include "mpi/zinoviev_a_sum_cols_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<int> zinoviev_a_sum_cols_matrix_mpi::generateRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

int zinoviev_a_sum_cols_matrix_mpi::computeLinearCoordinates(int x, int y, int width) { return y * width + x; }

std::vector<int> zinoviev_a_sum_cols_matrix_mpi::calculateMatrixSumSequential(const std::vector<int>& matrix, int width,
                                                                              int height, int startX, int endX) {
  std::vector<int> sums;

  for (int x = startX; x < endX; ++x) {
    int columnSum = 0;
    for (int y = 0; y < height; ++y) {
      int linearIdx = computeLinearCoordinates(x, y, width);
      columnSum += matrix[linearIdx];
    }
    sums.push_back(columnSum);
  }
  return sums;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Initialize the input and output vectors
  inputData_.resize(taskData->inputs_count[0]);
  auto* sourcePtr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned int i = 0; i < taskData->inputs_count[0]; ++i) {
    inputData_[i] = sourcePtr[i];
  }
  numCols = taskData->inputs_count[1];
  numRows = taskData->inputs_count[2];
  resultData_.resize(numCols, 0);

  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Validate the number of output elements
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  resultData_ = calculateMatrixSumSequential(inputData_, numCols, numRows, 0, numCols);
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < numCols; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = resultData_[i];
  }
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (mpiWorld.rank() == 0) {
    numRows = taskData->inputs_count[2];
    numCols = taskData->inputs_count[1];
  }

  broadcast(mpiWorld, numCols, 0);
  broadcast(mpiWorld, numRows, 0);

  if (mpiWorld.rank() == 0) {
    // Initialize the input vector
    inputData_ = std::vector<int>(taskData->inputs_count[0]);
    auto* temp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      inputData_[i] = temp_ptr[i];
    }
  } else {
    inputData_ = std::vector<int>(numCols * numRows);
  }

  broadcast(mpiWorld, inputData_.data(), numCols * numRows, 0);
  // Prepare the output vector
  resultData_ = std::vector<int>(numCols, 0);
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (mpiWorld.rank() == 0) {
    // Ensure output count matches expected size
    return taskData->outputs_count[0] == taskData->inputs_count[1];
  }
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int sizePerTask = numCols / mpiWorld.size();
  sizePerTask += (numCols % mpiWorld.size() == 0) ? 0 : 1;
  int lastColumn = std::min(numCols, sizePerTask * (mpiWorld.rank() + 1));
  auto localSums =
      calculateMatrixSumSequential(inputData_, numCols, numRows, sizePerTask * mpiWorld.rank(), lastColumn);

  localSums.resize(sizePerTask);

  if (mpiWorld.rank() == 0) {
    std::vector<int> accumulatedResults(numCols + sizePerTask * mpiWorld.size());
    std::vector<int> workerSizes(mpiWorld.size(), sizePerTask);
    boost::mpi::gatherv(mpiWorld, localSums.data(), localSums.size(), accumulatedResults.data(), workerSizes, 0);
    accumulatedResults.resize(numCols);
    resultData_ = accumulatedResults;
  } else {
    boost::mpi::gatherv(mpiWorld, localSums.data(), localSums.size(), 0);
  }
  return true;
}

bool zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (mpiWorld.rank() == 0) {
    for (int i = 0; i < numCols; ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = resultData_[i];
    }
  }
  return true;
}
