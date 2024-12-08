#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  numCols = taskData->inputs_count[1];
  numRows = taskData->inputs_count[2];

  inputMatrix_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmpPtr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    inputMatrix_[i] = tmpPtr[i];
  }

  result_ = std::vector<int>(numCols, 0);

  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[1] == 0 || taskData->inputs_count[2] == 0) {
    return false;
  }
  if (taskData->inputs.empty() || taskData->inputs_count[0] <= 0) {
    return false;
  }
  if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < numCols; j++) {
    auto maxElement = *std::max_element(inputMatrix_.begin() + j * numRows, inputMatrix_.begin() + (j + 1) * numRows);
    result_[j] = maxElement;
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < numCols; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = result_[i];
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    numCols = taskData->inputs_count[1];
    numRows = taskData->inputs_count[2];

    inputMatrix_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmpPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      inputMatrix_[i] = tmpPtr[i];
    }

    delta = numCols / world.size();
    extra = numCols % world.size();
  }

  result_ = std::vector<int>(numCols, 0);

  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[1] == 0 || taskData->inputs_count[2] == 0) {
      return false;
    }
    if (taskData->inputs.empty() || taskData->inputs_count[0] <= 0) {
      return false;
    }
    if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    numCols = taskData->inputs_count[1];
    numRows = taskData->inputs_count[2];
    delta = numCols / world.size();
    extra = numCols % world.size();
  }

  broadcast(world, numCols, 0);
  broadcast(world, numRows, 0);
  broadcast(world, delta, 0);
  broadcast(world, extra, 0);

  if (world.rank() == 0) {
    inputMatrix_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmpPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      inputMatrix_[i] = tmpPtr[i];
    }
  } else {
    inputMatrix_.resize(numCols * numRows, 0);
  }
  result_ = std::vector<int>(numCols, 0);

  broadcast(world, inputMatrix_.data(), numCols * numRows, 0);

  int startCol = delta * world.rank();
  int lastCol = startCol + delta;
  if (world.rank() == world.size() - 1) {
    lastCol += extra;
  }

  std::vector<int> localMax(lastCol - startCol);
  for (int j = startCol; j < lastCol; j++) {
    auto maxElem = *std::max_element(inputMatrix_.begin() + j * numRows, inputMatrix_.begin() + (j + 1) * numRows);
    localMax[j - startCol] = maxElem;
  }

  if (world.rank() == 0) {
    std::vector<int> globalRes(numCols);
    std::vector<int> sizes(world.size());
    for (int i = 0; i < world.size(); i++) {
      sizes[i] = delta + (i == world.size() - 1 ? extra : 0);
    }
    boost::mpi::gatherv(world, localMax.data(), localMax.size(), globalRes.data(), sizes, 0);
    result_ = globalRes;
  } else {
    boost::mpi::gatherv(world, localMax.data(), localMax.size(), 0);
  }

  return true;
}
bool nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < numCols; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = result_[i];
    }
  }
  return true;
}
