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

  column_per_proc = column_count / world.size();
  int remaining_columns = column_count % world.size();

  start_current_column = column_per_proc * world.rank() + std::min(world.rank(), remaining_columns);
  end_current_column = start_current_column + column_per_proc + (world.rank() < remaining_columns ? 1 : 0);
  int local_columns = end_current_column - start_current_column;

  input_.resize(row_count * local_columns);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      int proc_start_col = column_per_proc * proc + std::min(proc, remaining_columns);
      int proc_end_col = proc_start_col + column_per_proc + (proc < remaining_columns ? 1 : 0);
      int proc_columns = proc_end_col - proc_start_col;
      std::vector<int> columns_data(row_count * proc_columns);
      for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < proc_columns; ++j) {
          columns_data[i * proc_columns + j] = input_[i * column_count + proc_start_col + j];
        }
      }
      world.send(proc, 0, columns_data.data(), row_count * proc_columns);
    }
    for (int i = 0; i < row_count; ++i) {
      for (int j = 0; j < local_columns; ++j) {
        input_[i * local_columns + j] = input_[i * column_count + start_current_column + j];
      }
    }
  } else {
    world.recv(0, 0, input_.data(), row_count * local_columns);
  }
  std::vector<int> Max_on_proc(local_columns, std::numeric_limits<int>::min());
  for (int j = 0; j < local_columns; ++j) {
    for (int i = 0; i < row_count; ++i) {
      Max_on_proc[j] = std::max(Max_on_proc[j], input_[i * local_columns + j]);
    }
  }
  if (world.rank() == 0) {
    gathered_max_columns.resize(column_count);
    columns_per_process_count.resize(world.size());
    std::vector<int> displs(world.size());
    for (int i = 0; i < world.size(); ++i) {
      columns_per_process_count[i] = column_per_proc + (i < remaining_columns ? 1 : 0);
      displs[i] = (i == 0) ? 0 : displs[i - 1] + columns_per_process_count[i - 1];
    }
    boost::mpi::gatherv(world, Max_on_proc.data(), local_columns, gathered_max_columns.data(),
                        columns_per_process_count, displs, 0);
    res = gathered_max_columns;
  } else {
    boost::mpi::gatherv(world, Max_on_proc.data(), local_columns, 0);
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