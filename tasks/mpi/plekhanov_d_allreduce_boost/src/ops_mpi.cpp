#include "mpi/plekhanov_d_allreduce_boost/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

std::vector<int> plekhanov_d_allreduce_boost_mpi::getRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    int value = 0;
    do {
      value = gen() % 1000 + 1;
    } while (value <= 0);
    vec[i] = value;
  }
  return vec;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors

  columnCount = taskData->inputs_count[1];
  rowCount = taskData->inputs_count[2];

  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  inputData_.assign(tempPtr, tempPtr + taskData->inputs_count[0]);

  resultData_ = std::vector<int>(columnCount, 0);
  countAboveMin_ = std::vector<int>(columnCount, 0);

  return true;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[1] != 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
          taskData->inputs_count[0] > 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int column = 0; column < columnCount; column++) {
    int columnMin = inputData_[column];
    for (int row = 1; row < rowCount; row++) {
      if (inputData_[row * columnCount + column] < columnMin) {
        columnMin = inputData_[row * columnCount + column];
      }
    }
    resultData_[column] = columnMin;
  }
  for (int column = 0; column < columnCount; column++) {
    for (int row = 0; row < rowCount; row++) {
      if (inputData_[row * columnCount + column] > resultData_[column]) {
        countAboveMin_[column]++;
      }
    }
  }
  return true;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < columnCount; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = countAboveMin_[i];
  }
  return true;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskBoostParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    columnCount = taskData->inputs_count[1];
    rowCount = taskData->inputs_count[2];
  }

  if (world.rank() == 0) {
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    inputData_.assign(tempPtr, tempPtr + taskData->inputs_count[0]);
  } else {
    inputData_ = std::vector<int>(columnCount * rowCount, 0);
  }

  return true;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskBoostParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[1] != 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
            taskData->inputs_count[0] > 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
  }
  return true;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskBoostParallel::run() {
  internal_order_test();

  broadcast(world, columnCount, 0);
  broadcast(world, rowCount, 0);

  if (world.rank() != 0) {
    inputData_ = std::vector<int>(columnCount * rowCount, 0);
  }
  broadcast(world, inputData_.data(), columnCount * rowCount, 0);

  int delta = columnCount / world.size();
  int extra = columnCount % world.size();
  if (extra != 0) {
    delta += 1;
  }
  int startColumn = delta * world.rank();
  int lastColumn = std::min(columnCount, delta * (world.rank() + 1));
  std::vector<int> localMin(columnCount, INT_MAX);
  for (int column = startColumn; column < lastColumn; column++) {
    int minElem = inputData_[column];
    for (int row = 1; row < rowCount; row++) {
      int coordinate = row * columnCount + column;
      if (inputData_[coordinate] < minElem) {
        minElem = inputData_[coordinate];
      }
    }
    localMin[column] = minElem;
  }

  resultData_.resize(columnCount, 0);
  boost::mpi::all_reduce(world, localMin.data(), columnCount, resultData_.data(), boost::mpi::minimum<int>());

  std::vector<int> localCountAboveMin(columnCount, 0);
  for (int column = startColumn; column < lastColumn; column++) {
    for (int row = 0; row < rowCount; row++) {
      int coordinate = row * columnCount + column;
      if (inputData_[coordinate] > resultData_[column]) {
        localCountAboveMin[column]++;
      }
    }
  }

  countAboveMin_.resize(columnCount, 0);
  boost::mpi::reduce(world, localCountAboveMin.data(), columnCount, countAboveMin_.data(), std::plus<>(), 0);

  return true;
}

bool plekhanov_d_allreduce_boost_mpi::TestMPITaskBoostParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < columnCount; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = countAboveMin_[i];
    }
  }
  return true;
}