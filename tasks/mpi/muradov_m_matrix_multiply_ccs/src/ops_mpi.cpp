#include "mpi/muradov_m_matrix_multiply_ccs/include/ops_mpi.hpp"

#include <cassert>

namespace muradov_m_matrix_multiply_ccs_mpi {

bool MatrixMultiplyCCS::validation() {
  internal_order_test();

  int val_rows_A = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_cols_A = *reinterpret_cast<int*>(taskData->inputs[1]);
  int val_rows_B = *reinterpret_cast<int*>(taskData->inputs[2]);
  int val_cols_B = *reinterpret_cast<int*>(taskData->inputs[3]);

  return val_rows_A > 0 && val_cols_A > 0 && val_rows_B > 0 && val_cols_B > 0 && val_cols_A == val_rows_B;
}

bool MatrixMultiplyCCS::pre_processing() {
  internal_order_test();

  rows_A = *reinterpret_cast<int*>(taskData->inputs[0]);
  cols_A = *reinterpret_cast<int*>(taskData->inputs[1]);
  rows_B = *reinterpret_cast<int*>(taskData->inputs[2]);
  cols_B = *reinterpret_cast<int*>(taskData->inputs[3]);

  if (world.rank() == 0) {
    auto* A_val_ptr = reinterpret_cast<double*>(taskData->inputs[4]);
    int A_val_size = taskData->inputs_count[4];
    A_val.assign(A_val_ptr, A_val_ptr + A_val_size);

    auto* A_row_ind_ptr = reinterpret_cast<int*>(taskData->inputs[5]);
    int A_row_ind_size = taskData->inputs_count[5];
    A_row_ind.assign(A_row_ind_ptr, A_row_ind_ptr + A_row_ind_size);

    auto* A_col_ptr_ptr = reinterpret_cast<int*>(taskData->inputs[6]);
    int A_col_ptr_size = taskData->inputs_count[6];
    A_col_ptr.assign(A_col_ptr_ptr, A_col_ptr_ptr + A_col_ptr_size);

    auto* B_val_ptr = reinterpret_cast<double*>(taskData->inputs[7]);
    int B_val_size = taskData->inputs_count[7];
    B_val.assign(B_val_ptr, B_val_ptr + B_val_size);

    auto* B_row_ind_ptr = reinterpret_cast<int*>(taskData->inputs[8]);
    int B_row_ind_size = taskData->inputs_count[8];
    B_row_ind.assign(B_row_ind_ptr, B_row_ind_ptr + B_row_ind_size);

    auto* B_col_ptr_ptr = reinterpret_cast<int*>(taskData->inputs[9]);
    int B_col_ptr_size = taskData->inputs_count[9];
    B_col_ptr.assign(B_col_ptr_ptr, B_col_ptr_ptr + B_col_ptr_size);

    transpose_CCS(A_val, A_row_ind, A_col_ptr, rows_A, cols_A, At_val, At_row_ind, At_col_ptr);

    rows_At = cols_A;
    cols_At = rows_A;
  }

  return true;
}

bool MatrixMultiplyCCS::run() {
  internal_order_test();

  color = static_cast<int>(world.rank() < cols_B);
  comm = world.split(color);

  boost::mpi::broadcast(comm, B_val, 0);
  boost::mpi::broadcast(comm, B_row_ind, 0);
  boost::mpi::broadcast(comm, B_col_ptr, 0);

  boost::mpi::broadcast(comm, At_val, 0);
  boost::mpi::broadcast(comm, At_row_ind, 0);
  boost::mpi::broadcast(comm, At_col_ptr, 0);
  boost::mpi::broadcast(comm, rows_At, 0);

  if (color == 1) {
    auto pair = split_into_segments(cols_B, comm.size(), comm.rank());

    loc_start = pair.first;
    loc_end = pair.second;
    loc_cols = loc_end - loc_start;

    extract_columns(B_val, B_row_ind, B_col_ptr, loc_start, loc_end, loc_val, loc_row_ind, loc_col_ptr);

    multiply_CCS(At_val, At_row_ind, At_col_ptr, rows_At, loc_val, loc_row_ind, loc_col_ptr, loc_cols, loc_res_val,
                 loc_res_row_ind, loc_res_col_ptr);

    std::vector<int> sizes_val;
    if (comm.rank() == 0) {
      sizes_val.resize(comm.size());
    }
    boost::mpi::gather(comm, loc_res_col_ptr.back(), sizes_val.data(), 0);

    int size_ptr_vector = 0;
    boost::mpi::reduce(comm, static_cast<int>(loc_res_col_ptr.size() - 1), size_ptr_vector, std::plus<>(), 0);

    std::vector<int> sizes_ptr;
    if (comm.rank() == 0) {
      sizes_ptr.resize(size_ptr_vector);
    }
    boost::mpi::gather(comm, static_cast<int>(loc_res_col_ptr.size() - 1), sizes_ptr, 0);

    if (comm.rank() == 0) {
      int sum = std::accumulate(sizes_val.begin(), sizes_val.end(), 0);
      res_val.resize(sum);
      res_ind.resize(sum);
      res_ptr.resize(size_ptr_vector);

      boost::mpi::gatherv(comm, loc_res_val.data(), loc_res_val.size(), res_val.data(), sizes_val, 0);
      boost::mpi::gatherv(comm, loc_res_row_ind.data(), loc_res_row_ind.size(), res_ind.data(), sizes_val, 0);
      boost::mpi::gatherv(comm, loc_res_col_ptr.data(), loc_res_col_ptr.size() - 1, res_ptr.data(), sizes_ptr, 0);

      int shift = 0;
      int offset = 0;
      for (size_t j = 0; j < sizes_ptr.size(); j++) {
        shift = sizes_val[j];
        offset += sizes_ptr[j];
        for (size_t i = offset; i < res_ptr.size(); i++) {
          res_ptr[i] += shift;
        }
      }

      res_ptr.push_back(sum);
    } else {
      boost::mpi::gatherv(comm, loc_res_val.data(), loc_res_val.size(), 0);
      boost::mpi::gatherv(comm, loc_res_row_ind.data(), loc_res_row_ind.size(), 0);
      boost::mpi::gatherv(comm, loc_res_col_ptr.data(), loc_res_col_ptr.size() - 1, 0);
    }
  }

  return true;
}

bool MatrixMultiplyCCS::post_processing() {
  internal_order_test();

  if (color == 1 && comm.rank() == 0) {
    auto* C_val_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    auto* C_row_ind_ptr = reinterpret_cast<int*>(taskData->outputs[1]);
    auto* C_col_ptr_ptr = reinterpret_cast<int*>(taskData->outputs[2]);

    std::copy(res_val.begin(), res_val.end(), C_val_ptr);
    std::copy(res_ind.begin(), res_ind.end(), C_row_ind_ptr);
    std::copy(res_ptr.begin(), res_ptr.end(), C_col_ptr_ptr);
  }

  return true;
}

}  // namespace muradov_m_matrix_multiply_ccs_mpi
