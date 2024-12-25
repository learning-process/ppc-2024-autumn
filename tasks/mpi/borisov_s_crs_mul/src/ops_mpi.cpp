#include "mpi/borisov_s_crs_mul/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <vector>

using namespace std::chrono_literals;

namespace borisov_s_crs_mul_mpi {

bool CrsMatrixMulTaskMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    A_values_.assign(reinterpret_cast<const double*>(taskData->inputs[0]),
                     reinterpret_cast<const double*>(taskData->inputs[0]) + taskData->inputs_count[0]);
    A_col_index_.assign(reinterpret_cast<const int*>(taskData->inputs[1]),
                        reinterpret_cast<const int*>(taskData->inputs[1]) + taskData->inputs_count[1]);
    A_row_ptr_.assign(reinterpret_cast<const int*>(taskData->inputs[2]),
                      reinterpret_cast<const int*>(taskData->inputs[2]) + taskData->inputs_count[2]);

    B_values_.assign(reinterpret_cast<const double*>(taskData->inputs[3]),
                     reinterpret_cast<const double*>(taskData->inputs[3]) + taskData->inputs_count[3]);
    B_col_index_.assign(reinterpret_cast<const int*>(taskData->inputs[4]),
                        reinterpret_cast<const int*>(taskData->inputs[4]) + taskData->inputs_count[4]);
    B_row_ptr_.assign(reinterpret_cast<const int*>(taskData->inputs[5]),
                      reinterpret_cast<const int*>(taskData->inputs[5]) + taskData->inputs_count[5]);

    A_nrows_ = static_cast<int>(taskData->inputs_count[2] - 1);
    B_nrows_ = static_cast<int>(taskData->inputs_count[5] - 1);
    A_ncols_ = *std::max_element(A_col_index_.begin(), A_col_index_.end()) + 1;
    B_ncols_ = *std::max_element(B_col_index_.begin(), B_col_index_.end()) + 1;

    C_nrows_ = A_nrows_;
    C_ncols_ = B_ncols_;
    C_values_.clear();
    C_col_index_.clear();
    C_row_ptr_.assign(C_nrows_ + 1, 0);
  }

  return true;
}

bool CrsMatrixMulTaskMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.size() != 6 || taskData->outputs.size() != 3) {
      return false;
    }

    if (taskData->inputs_count.size() < 6 || taskData->outputs_count.size() < 3) {
      return false;
    }

    int A_ncols = *std::max_element(reinterpret_cast<const int*>(taskData->inputs[1]),
                                    reinterpret_cast<const int*>(taskData->inputs[1]) + taskData->inputs_count[1]) +
                  1;

    int B_nrows = static_cast<int>(taskData->inputs_count[5] - 1);

    if (A_ncols != B_nrows) {
      return false;
    }
  }

  return true;
}

bool CrsMatrixMulTaskMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, A_values_, 0);
  boost::mpi::broadcast(world, A_col_index_, 0);
  boost::mpi::broadcast(world, A_row_ptr_, 0);
  boost::mpi::broadcast(world, B_values_, 0);
  boost::mpi::broadcast(world, B_col_index_, 0);
  boost::mpi::broadcast(world, B_row_ptr_, 0);
  boost::mpi::broadcast(world, C_nrows_, 0);
  boost::mpi::broadcast(world, C_ncols_, 0);

  int rows_per_process = C_nrows_ / world.size();
  int start_row = world.rank() * rows_per_process;
  int end_row = (world.rank() == world.size() - 1) ? C_nrows_ : start_row + rows_per_process;

  std::vector<double> temp(C_ncols_, 0.0);
  std::vector<double> local_values;
  std::vector<int> local_col_index;
  std::vector<int> local_row_ptr(end_row - start_row + 1, 0);

  int local_nnz_count = 0;

  for (int i = start_row; i < end_row; ++i) {
    std::fill(temp.begin(), temp.end(), 0.0);

    for (int posA = A_row_ptr_[i]; posA < A_row_ptr_[i + 1]; ++posA) {
      double a_val = A_values_[posA];
      int a_col = A_col_index_[posA];
      for (int posB = B_row_ptr_[a_col]; posB < B_row_ptr_[a_col + 1]; ++posB) {
        int b_col = B_col_index_[posB];
        double b_val = B_values_[posB];
        temp[b_col] += a_val * b_val;
      }
    }

    local_row_ptr[i - start_row] = local_nnz_count;
    for (int col = 0; col < C_ncols_; ++col) {
      if (temp[col] != 0.0) {
        local_values.push_back(temp[col]);
        local_col_index.push_back(col);
        ++local_nnz_count;
      }
    }
  }

  local_row_ptr[end_row - start_row] = local_nnz_count;

  if (world.rank() == 0) {
    std::vector<std::vector<double>> all_values(world.size());
    std::vector<std::vector<int>> all_col_indices(world.size());
    std::vector<std::vector<int>> all_row_ptrs(world.size());

    boost::mpi::gather(world, local_values, all_values, 0);
    boost::mpi::gather(world, local_col_index, all_col_indices, 0);
    boost::mpi::gather(world, local_row_ptr, all_row_ptrs, 0);

    for (int i = 0; i < world.size(); ++i) {
      C_values_.insert(C_values_.end(), all_values[i].begin(), all_values[i].end());
      C_col_index_.insert(C_col_index_.end(), all_col_indices[i].begin(), all_col_indices[i].end());

      if (i == 0) {
        C_row_ptr_ = all_row_ptrs[i];
      } else {
        int offset = C_row_ptr_.back();
        for (size_t j = 1; j < all_row_ptrs[i].size(); ++j) {
          C_row_ptr_.push_back(all_row_ptrs[i][j] + offset);
        }
      }
    }
  } else {
    boost::mpi::gather(world, local_values, 0);
    boost::mpi::gather(world, local_col_index, 0);
    boost::mpi::gather(world, local_row_ptr, 0);
  }

  return true;
}

bool CrsMatrixMulTaskMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    taskData->outputs_count[0] = C_values_.size();
    taskData->outputs_count[1] = C_col_index_.size();
    taskData->outputs_count[2] = C_row_ptr_.size();

    taskData->outputs[0] = reinterpret_cast<uint8_t*>(C_values_.data());
    taskData->outputs[1] = reinterpret_cast<uint8_t*>(C_col_index_.data());
    taskData->outputs[2] = reinterpret_cast<uint8_t*>(C_row_ptr_.data());
  }

  return true;
}

}  // namespace borisov_s_crs_mul_mpi