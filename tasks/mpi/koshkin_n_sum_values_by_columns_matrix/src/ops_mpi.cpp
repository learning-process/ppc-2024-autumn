#include "mpi/koshkin_n_sum_values_by_columns_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  rows = taskData->inputs_count[0];
  columns = taskData->inputs_count[1];

  // TaskData
  input_.resize(rows, std::vector<int>(columns));

  int* inputMatrix = reinterpret_cast<int*>(taskData->inputs[0]);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      input_[i][j] = inputMatrix[i * columns + j];
    }
  }

  res.resize(columns, 0);  // sumColumns
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < columns; ++j) {
    res[j] = 0;
    for (int i = 0; i < rows; ++i) {
      res[j] += input_[i][j];
    }
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int* outputSums = reinterpret_cast<int*>(taskData->outputs[0]);
  for (int j = 0; j < columns; ++j) {
    outputSums[j] = res[j];
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];
    input_.resize(rows * columns);
    int* inputMatrix = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        input_[i * columns + j] = inputMatrix[i * columns + j];
      }
    }
  }

  res.resize(columns, 0);
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
            taskData->inputs_count[1] == taskData->outputs_count[0]);
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];
  }

  broadcast(world, rows, 0);
  broadcast(world, columns, 0);

  int rows_per_process;
  int extra_rows;

  if (world.rank() == 0) {
    rows_per_process = rows / world.size();
    extra_rows = rows % world.size();
  }

  broadcast(world, rows_per_process, 0);
  broadcast(world, extra_rows, 0);

  int local_rows = rows_per_process + (world.rank() < extra_rows ? 1 : 0);

  local_input_.resize(local_rows * columns);

  if (world.rank() == 0) {
    int offset = local_rows * columns;
    for (int proc = 1; proc < world.size(); ++proc) {
      int proc_rows = rows_per_process + (proc < extra_rows ? 1 : 0);
      world.send(proc, 2, input_.data() + offset, proc_rows * columns);
      offset += proc_rows * columns;
    }
    std::copy(input_.begin(), input_.begin() + local_rows * columns, local_input_.begin());
  } else {
    world.recv(0, 2, local_input_.data(), local_rows * columns);
  }

  std::vector<int> local_sum(columns, 0);

  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      local_sum[j] += local_input_[i * columns + j];
    }
  }

  res.resize(columns, 0);
  boost::mpi::reduce(world, local_sum.data(), columns, res.data(), std::plus<>(), 0);

  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* outputSums = reinterpret_cast<int*>(taskData->outputs[0]);
    for (int j = 0; j < columns; ++j) {
      outputSums[j] = res[j];
    }
  }
  return true;
}