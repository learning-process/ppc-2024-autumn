#include "mpi/kapustin_i_max_cols/include/avg_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI::pre_processing() {
  internal_order_test();
  column_count = *reinterpret_cast<int*>(taskData->inputs[1]);
  int total_elements = taskData->inputs_count[0];
  row_count = total_elements / column_count;
  input_.resize(total_elements);
  auto* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(matrix_data, matrix_data + total_elements);
  res.resize(column_count, std::numeric_limits<int>::min());
  return true;
}
bool kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI::validation() {
  internal_order_test();
  return (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0);
}
bool kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI::run() {
  {
    internal_order_test();
    for (int j = 0; j < column_count; ++j) {
      int max_value = std::numeric_limits<int>::min();
      for (int i = 0; i < row_count; ++i) {
        int current_value = input_[i * column_count + j];
        if (current_value > max_value) {
          max_value = current_value;
        }
      }
      res[j] = max_value;
    }
    return true;
  }
}
bool kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI::post_processing() {
  internal_order_test();
  for (int j = 0; j < column_count; ++j) {
    reinterpret_cast<int*>(taskData->outputs[0])[j] = res[j];
  }
  return true;
}

bool kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    column_count = taskData->inputs_count[1];
    row_count = taskData->inputs_count[2];
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = matrix_data[i];
    }
  }
  res.assign(column_count, std::numeric_limits<int>::min());
  return true;
}
bool kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == taskData->inputs_count[1]);
  }
  return true;
}
bool kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI::run() {
  internal_order_test();

  broadcast(world, column_count, 0);
  broadcast(world, row_count, 0);

  if (world.rank() != 0) {
    input_.assign(column_count * row_count, 0);
  }

  broadcast(world, input_.data(), column_count * row_count, 0);

  column_per_proc = column_count / world.size() + (column_count % world.size() != 0 ? 1 : 0);
  start_current_column = column_per_proc * world.rank();
  end_current_column = std::min(column_count, column_per_proc * (world.rank() + 1));
  std::vector<int> Max_on_proc;
  Max_on_proc.reserve(column_per_proc);
  for (int j = start_current_column; j < end_current_column; j++) {
    if (j < column_count) {
      int founded_max_element = input_[j];
      for (int i = 1; i < row_count; i++) {
        int idx = i * column_count + j;
        if (idx < static_cast<int>(input_.size())) {
          if (input_[idx] > founded_max_element) {
            founded_max_element = input_[idx];
          }
        }
      }
      Max_on_proc.push_back(founded_max_element);
    }
  }

  Max_on_proc.resize(column_per_proc = column_count / world.size() + (column_count % world.size() != 0 ? 1 : 0));

  if (world.rank() == 0) {
    gathered_max_columns.resize(column_count + column_per_proc * world.size());
    columns_per_process_count = std::vector<int>(world.size(), column_per_proc);
    boost::mpi::gatherv(world, Max_on_proc.data(), Max_on_proc.size(), gathered_max_columns.data(),
                        columns_per_process_count, 0);
    gathered_max_columns.resize(column_count);
    res = gathered_max_columns;
  } else {
    boost::mpi::gatherv(world, Max_on_proc.data(), Max_on_proc.size(), 0);
  }
  return true;
}
bool kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < column_count; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}