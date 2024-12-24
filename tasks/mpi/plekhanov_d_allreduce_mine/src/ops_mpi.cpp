#include "mpi/plekhanov_d_allreduce_mine/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  columnCount = taskData->inputs_count[1];
  rowCount = taskData->inputs_count[2];

  auto *tempPtr = reinterpret_cast<int *>(taskData->inputs[0]);
  inputData_.assign(tempPtr, tempPtr + taskData->inputs_count[0]);

  resultData_.resize(columnCount, 0);
  countAboveMin_.resize(columnCount, 0);

  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[1] != 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
          taskData->inputs_count[0] > 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int column = 0; column < columnCount; column++) {
    int columnMin = inputData_[column];
    for (int row = 1; row < rowCount; row++) {
      int value = inputData_[row * columnCount + column];
      if (value < columnMin) {
        columnMin = value;
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

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < columnCount; i++) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = countAboveMin_[i];
  }
  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    columnCount = taskData->inputs_count[1];
    rowCount = taskData->inputs_count[2];
  }

  if (world.rank() == 0) {
    // init vectors
    auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    inputData_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  } else {
    inputData_ = std::vector<int>(columnCount * rowCount, 0);
  }

  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[1] != 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
            taskData->inputs_count[0] > 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
  }
  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel::run() {
  internal_order_test();

  broadcast(world, rowCount, 0);
  broadcast(world, columnCount, 0);

  int lambda_1 = rowCount / world.size();
  int lambda_2 = rowCount % world.size();

  broadcast(world, lambda_1, 0);
  broadcast(world, lambda_2, 0);

  std::vector<int> size(world.size(), (lambda_1 * columnCount));
  for (int i = 0; i < lambda_2; i++) size[world.size() - i - 1] += columnCount;

  localInputData_.resize(size[world.rank()]);
  scatterv(world, inputData_, size, localInputData_.data(), 0);

  std::vector<int> min_by_cols(columnCount, std::numeric_limits<int>::max());
  std::vector<int> local_min_by_cols(columnCount, std::numeric_limits<int>::max());
  count_greater.resize(columnCount, 0);
  std::vector<int> local_count_greater(columnCount, 0);

  if (!localInputData_.empty()) {
    for (size_t j = 0; j < localInputData_.size() / columnCount; j++) {
      for (int i = 0; i < columnCount; i++) {
        int value = localInputData_[j * columnCount + i];
        if (value < local_min_by_cols[i]) local_min_by_cols[i] = value;
      }
    }
  }
  boost::mpi::all_reduce(world, local_min_by_cols.data(), columnCount, min_by_cols.data(), boost::mpi::minimum<int>());

  if (!localInputData_.empty()) {
    for (size_t j = 0; j < localInputData_.size() / columnCount; j++) {
      for (int i = 0; i < columnCount; i++) {
        if (localInputData_[j * columnCount + i] > min_by_cols[i]) {
          local_count_greater[i]++;
        }
      }
    }
  }
  boost::mpi::all_reduce(world, local_count_greater.data(), columnCount, count_greater.data(), std::plus<>());

  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < columnCount; i++) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = count_greater[i];
    }
  }
  return true;
}

template <typename T>
void plekhanov_d_allreduce_mine_mpi::TestMPITaskBoostParallel::my_all_reduce(const boost::mpi::communicator &world,
                                                                             const T *in_values, T *out_values, int n) {
  int root = world.rank();
  std::vector<T> left_values(n);
  std::vector<T> right_values(n);

  int left_child = 2 * root + 1;
  int right_child = 2 * root + 2;

  std::copy(in_values, in_values + n, out_values);

  if (left_child < world.size()) {
    world.recv(left_child, 0, left_values.data(), n);
  }

  if (right_child < world.size()) {
    world.recv(right_child, 0, right_values.data(), n);
  }

  for (int i = 0; i < n; ++i) {
    if (left_child < world.size()) {
      out_values[i] = std::min(out_values[i], left_values[i]);
    }
    if (right_child < world.size()) {
      out_values[i] = std::min(out_values[i], right_values[i]);
    }
  }
  if (root != 0) {
    int parent = (root - 1) / 2;
    world.send(parent, 0, out_values, n);
    world.recv(parent, 0, out_values, n);
  }

  if (left_child < world.size()) {
    world.send(left_child, 0, out_values, n);
  }

  if (right_child < world.size()) {
    world.send(right_child, 0, out_values, n);
  }
}