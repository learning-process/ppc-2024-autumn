// Copyright 2024 Nesterov Alexander
// shlyakov_m_min_value_of_row
#include "mpi/shlyakov_m_ccs_mult_mpi/include/ops_mpi.hpp"

#include <thread>
#include <vector>

using namespace shlyakov_m_ccs_mult_mpi;

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

  int size = world.size();
  int rank = world.rank();

  bool single_element_A = (A_.values.size() == 1 && A_.col_pointers.size() == 2);
  bool single_element_B = (B_.values.size() == 1 && B_.col_pointers.size() == 2);

  SparseMatrix C_subset;
  C_subset.col_pointers.reserve(cols_b + 1);  
  C_subset.col_pointers.push_back(0); 

  if (single_element_A || single_element_B) {
    if (rank == 0) {
      C_subset.values.reserve(A_.values.size() * B_.values.size());
      C_subset.row_indices.reserve(A_.values.size() * B_.values.size());

      std::vector<double> Cj_map(rows_a, 0.0);

      for (int j = 0; j < cols_b; ++j) {
        std::fill(Cj_map.begin(), Cj_map.end(), 0.0);

        int Bj_start = B_.col_pointers[j];
        int Bj_end = B_.col_pointers[j + 1];

        for (int idx = Bj_start; idx < Bj_end; ++idx) {
          int k = B_.row_indices[idx];
          double Bkj = B_.values[idx];

          int Ak_start = A_.col_pointers[k];
          int Ak_end = A_.col_pointers[k + 1];

          for (int a_idx = Ak_start; a_idx < Ak_end; ++a_idx) {
            int row = A_.row_indices[a_idx];
            double Aik = A_.values[a_idx];
            Cj_map[row] += Aik * Bkj;
          }
        }

        for (int row = 0; row < rows_a; ++row) {
          double val = Cj_map[row];
          if (std::abs(val) > 1e-12) {
            C_subset.row_indices.push_back(row);
            C_subset.values.push_back(val);
          }
        }

        C_subset.col_pointers.push_back(C_subset.values.size());
      }

      result_ = C_subset;
    }
    return true;
  } else {

    boost::mpi::broadcast(world, A_.values, 0);
    boost::mpi::broadcast(world, A_.row_indices, 0);
    boost::mpi::broadcast(world, A_.col_pointers, 0);
    boost::mpi::broadcast(world, rows_a, 0);
    boost::mpi::broadcast(world, cols_a, 0);

    boost::mpi::broadcast(world, B_.values, 0);
    boost::mpi::broadcast(world, B_.row_indices, 0);
    boost::mpi::broadcast(world, B_.col_pointers, 0);
    boost::mpi::broadcast(world, rows_b, 0);
    boost::mpi::broadcast(world, cols_b, 0);

    int cols_per_proc = cols_b / size;
    int remainder = cols_b % size;

    int start_col, end_col;
    if (rank < remainder) {
      start_col = rank * (cols_per_proc + 1);
      end_col = start_col + cols_per_proc + 1;
    } else {
      start_col = rank * cols_per_proc + remainder;
      end_col = start_col + cols_per_proc;
    }

    SparseMatrix B_subset;
    B_subset.col_pointers.reserve(end_col - start_col + 1);
    B_subset.col_pointers.push_back(0);

    for (int col = start_col; col < end_col; ++col) {
      int col_start = B_.col_pointers[col];
      int col_end = B_.col_pointers[col + 1];

      B_subset.values.insert(B_subset.values.end(), B_.values.begin() + col_start, B_.values.begin() + col_end);
      B_subset.row_indices.insert(B_subset.row_indices.end(), B_.row_indices.begin() + col_start,
                                  B_.row_indices.begin() + col_end);
      B_subset.col_pointers.push_back(B_subset.values.size());
    }

    std::vector<double> Cj_map(rows_a, 0.0);

    for (int j = 0; j < (end_col - start_col); ++j) {
      std::fill(Cj_map.begin(), Cj_map.end(), 0.0);

      int Bj_start = B_subset.col_pointers[j];
      int Bj_end = B_subset.col_pointers[j + 1];

      for (int idx = Bj_start; idx < Bj_end; ++idx) {
        int k = B_subset.row_indices[idx];
        double Bkj = B_subset.values[idx];
        int Ak_start = A_.col_pointers[k];
        int Ak_end = A_.col_pointers[k + 1];

        for (int a_idx = Ak_start; a_idx < Ak_end; ++a_idx) {
          int row = A_.row_indices[a_idx];
          double Aik = A_.values[a_idx];
          Cj_map[row] += Aik * Bkj;
        }
      }

      for (int row = 0; row < rows_a; ++row) {
        double val = Cj_map[row];
        if (std::abs(val) > 1e-12) {
          C_subset.row_indices.push_back(row);
          C_subset.values.push_back(val);
        }
      }

      C_subset.col_pointers.push_back(C_subset.values.size());
    }

    std::vector<SparseMatrix> all_C_subsets;
    if (rank == 0) {
      all_C_subsets.resize(size);
    }

    boost::mpi::gather(world, C_subset, all_C_subsets, 0);

    if (rank == 0) {
      result_.values.reserve(cols_b * rows_a); 
      result_.row_indices.reserve(cols_b * rows_a);
      result_.col_pointers.reserve(cols_b + 1);
      result_.col_pointers.push_back(0);  

      for (int proc = 0; proc < size; ++proc) {
        SparseMatrix& C_proc = all_C_subsets[proc];
        int num_cols = C_proc.col_pointers.size() - 1;

        for (int j = 0; j < num_cols; ++j) {
          int col_start = C_proc.col_pointers[j];
          int col_end = C_proc.col_pointers[j + 1];

          result_.values.insert(result_.values.end(), C_proc.values.begin() + col_start,
                                C_proc.values.begin() + col_end);
          result_.row_indices.insert(result_.row_indices.end(), C_proc.row_indices.begin() + col_start,
                                     C_proc.row_indices.begin() + col_end);

          result_.col_pointers.push_back(result_.values.size());
        }
      }
    }

    return true;
  }
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