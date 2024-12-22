// Copyright 2024 Nesterov Alexander
// shlyakov_m_min_value_of_row
#include "mpi/shlyakov_m_ccs_mult_mpi/include/ops_mpi.hpp"

#include <thread>
#include <vector>

bool shlyakov_m_ccs_mult_mpi::TestTaskMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    const double* a_values = reinterpret_cast<const double*>(taskData->inputs[0]);
    const int* a_row_indices = reinterpret_cast<const int*>(taskData->inputs[1]);
    const int* a_col_pointers = reinterpret_cast<const int*>(taskData->inputs[2]);

    const double* b_values = reinterpret_cast<const double*>(taskData->inputs[3]);
    const int* b_row_indices = reinterpret_cast<const int*>(taskData->inputs[4]);
    const int* b_col_pointers = reinterpret_cast<const int*>(taskData->inputs[5]);

    A_.values.assign(a_values, a_values + taskData->inputs_count[0]);
    A_.row_indices.assign(a_row_indices, a_row_indices + taskData->inputs_count[1]);
    A_.col_pointers.assign(a_col_pointers, a_col_pointers + taskData->inputs_count[2] + 1);

    B_.values.assign(b_values, b_values + taskData->inputs_count[3]);
    B_.row_indices.assign(b_row_indices, b_row_indices + taskData->inputs_count[4]);
    B_.col_pointers.assign(b_col_pointers, b_col_pointers + taskData->inputs_count[5] + 1);

    rows_a = taskData->inputs_count[2];
    rows_b = taskData->inputs_count[5];
    cols_a = A_.col_pointers.size() - 1;
    cols_b = B_.col_pointers.size() - 1;
  }

  return true;
}

bool shlyakov_m_ccs_mult_mpi::TestTaskMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData == nullptr || taskData->inputs.size() != 6 || taskData->inputs_count.size() < 6 ||
        static_cast<int>(taskData->inputs_count[2]) < 0 || static_cast<int>(taskData->inputs_count[5]) < 0 ||
        static_cast<int>(taskData->inputs_count[0]) != static_cast<int>(taskData->inputs_count[1]) ||
        static_cast<int>(taskData->inputs_count[3]) != static_cast<int>(taskData->inputs_count[4]) ||
        (taskData->inputs_count[0] <= 0 && taskData->inputs_count[3] <= 0)) {
      return false;
    }
  }

  return true;
}

bool shlyakov_m_ccs_mult_mpi::TestTaskMPI::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  result_.col_pointers.clear();
  result_.values.clear();
  result_.row_indices.clear();
  result_.col_pointers.push_back(0);

  std::vector<double> temp(rows_a, 0.0);
  if (size == 1) {
    // Однопоточный режим
    for (int col_b = 0; col_b < cols_b; ++col_b) {
      std::fill(temp.begin(), temp.end(), 0.0);
      for (int k = 0; k < rows_a; ++k) {
        int a_start = A_.col_pointers[k];
        int a_end = A_.col_pointers[k + 1];
        for (int pos_a = a_start; pos_a < a_end; ++pos_a) {
          double a_val = A_.values[pos_a];
          int row_a = A_.row_indices[pos_a];

          int b_start = B_.col_pointers[col_b];
          int b_end = B_.col_pointers[col_b + 1];
          for (int pos_b = b_start; pos_b < b_end; ++pos_b) {
            double b_val = B_.values[pos_b];
            int row_b = B_.row_indices[pos_b];
            if (row_b == k) {
              temp[row_a] += a_val * b_val;
            }
          }
        }
      }
      for (int row_a = 0; row_a < rows_a; ++row_a) {
        if (temp[row_a] != 0.0) {
          result_.values.push_back(temp[row_a]);
          result_.row_indices.push_back(row_a);
        }
      }
      result_.col_pointers.push_back(result_.values.size());
    }

  } else {
    // Параллельный режим
    std::vector<double> local_result_values;
    std::vector<int> local_result_row_indices;
    std::vector<int> local_result_col_pointers;
    local_result_col_pointers.push_back(0);

    int cols_per_process = cols_b / (size - 1);
    int remainder = cols_b % (size - 1);

    if (rank == 0) {
      // Мастер-процесс
      for (int i = 1; i < size; ++i) {
        int start_col = (i - 1) * cols_per_process;
        int end_col = start_col + cols_per_process;
        if (i == size - 1) {
          end_col += remainder;
        }
        std::vector<int> cols_to_process;
        for (int col = start_col; col < end_col; ++col) {
          cols_to_process.push_back(col);
        }
        world.send(i, 0, cols_to_process);
      }
      for (int i = 1; i < size; ++i) {
        std::vector<double> recv_values;
        std::vector<int> recv_row_indices;
        std::vector<int> recv_col_pointers;

        world.recv(i, 1, recv_values);
        world.recv(i, 2, recv_row_indices);
        world.recv(i, 3, recv_col_pointers);

        result_.values.insert(result_.values.end(), recv_values.begin(), recv_values.end());
        result_.row_indices.insert(result_.row_indices.end(), recv_row_indices.begin(), recv_row_indices.end());

        for (size_t j = 1; j < recv_col_pointers.size(); ++j) {
          result_.col_pointers.push_back(result_.values.size() - (recv_values.size() - recv_col_pointers[j]));
        }
      }

    } else {
      // Рабочий процесс
      std::vector<int> cols_to_process;
      world.recv(0, 0, cols_to_process);
      for (int col_b : cols_to_process) {
        std::fill(temp.begin(), temp.end(), 0.0);
        for (int k = 0; k < rows_a; ++k) {
          int a_start = A_.col_pointers[k];
          int a_end = A_.col_pointers[k + 1];
          for (int pos_a = a_start; pos_a < a_end; ++pos_a) {
            double a_val = A_.values[pos_a];
            int row_a = A_.row_indices[pos_a];

            int b_start = B_.col_pointers[col_b];
            int b_end = B_.col_pointers[col_b + 1];
            for (int pos_b = b_start; pos_b < b_end; ++pos_b) {
              double b_val = B_.values[pos_b];
              int row_b = B_.row_indices[pos_b];
              if (row_b == k) {
                temp[row_a] += a_val * b_val;
              }
            }
          }
        }
        for (int row_a = 0; row_a < rows_a; ++row_a) {
          if (temp[row_a] != 0.0) {
            local_result_values.push_back(temp[row_a]);
            local_result_row_indices.push_back(row_a);
          }
        }
        local_result_col_pointers.push_back(local_result_values.size());
      }
      world.send(0, 1, local_result_values);
      world.send(0, 2, local_result_row_indices);
      world.send(0, 3, local_result_col_pointers);
    }
  }

  return true;
}

bool shlyakov_m_ccs_mult_mpi::TestTaskMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result_.values.data()));
    taskData->outputs_count.push_back(static_cast<unsigned int>(result_.values.size() * sizeof(double)));

    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result_.row_indices.data()));
    taskData->outputs_count.push_back(static_cast<unsigned int>(result_.row_indices.size() * sizeof(int)));

    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result_.col_pointers.data()));
    taskData->outputs_count.push_back(static_cast<unsigned int>(result_.col_pointers.size() * sizeof(int)));
  }
  return true;
}