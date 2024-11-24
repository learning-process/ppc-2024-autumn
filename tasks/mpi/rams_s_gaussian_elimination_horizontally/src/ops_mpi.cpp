#include "mpi/rams_s_gaussian_elimination_horizontally/include/ops_mpi.hpp"

#include <algorithm>
#include <vector>

#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"

using namespace std::chrono_literals;

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  auto *input_data = reinterpret_cast<double *>(taskData->inputs[0]);
  matrix = std::vector<double>(input_data, input_data + taskData->inputs_count[0]);
  cols_count = taskData->outputs_count[0] + 1;
  rows_count = matrix.size() / cols_count;
  res = std::vector<double>(taskData->outputs_count[0], std::numeric_limits<double>::quiet_NaN());
  return true;
}

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] >= 0 &&
         (taskData->inputs_count[0] % (taskData->outputs_count[0] + 1) == 0);
}

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  std::vector<int> swapped_rows(rows_count);

  for (int i = 0; i < rows_count; i++) {
    swapped_rows[i] = i;
  }

  auto get_item_ptr = [&](int row, int col) { return &matrix[swapped_rows[row] * cols_count + col]; };
  auto get_item = [&](int row, int col) { return *get_item_ptr(row, col); };

  // forward elimination
  for (int current_row = 0; current_row < rows_count; current_row++) {
    int pivot_row = rows_count - 1;
    int pivot_col = cols_count;
    for (int row = current_row; row < rows_count; row++) {
      for (int col = current_row; col < cols_count; col++) {
        if (get_item(row, col) != 0) {
          if (col < pivot_col) {
            pivot_row = row;
            pivot_col = col;
          }
          break;
        }
      }
    }
    std::swap(swapped_rows[current_row], swapped_rows[pivot_row]);

    for (int row = current_row + 1; row < rows_count; row++) {
      double ratio = get_item(row, pivot_col) / get_item(current_row, pivot_col);
      for (int col = pivot_col; col < cols_count; col++) {
        *get_item_ptr(row, col) -= get_item(current_row, +col) * ratio;
      }
    }
  }

  // back substitution
  for (int current_row = rows_count - 1; current_row >= 0; current_row--) {
    for (int col = 0; col < cols_count - 1; col++) {
      if (get_item(current_row, col) != 0) {
        double known_part = 0;
        for (int variable_col = col + 1; variable_col < cols_count - 1; variable_col++) {
          if (get_item(current_row, variable_col) != 0) {
            known_part += res[variable_col] * get_item(current_row, variable_col);
          }
        }

        res[col] = -(get_item(current_row, cols_count - 1) + known_part) / get_item(current_row, col);

        break;
      }
    }
  }
  return true;
}

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  std::copy(res.begin(), res.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}

/////

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto *input_data = reinterpret_cast<double *>(taskData->inputs[0]);
    matrix = std::vector<double>(input_data, input_data + taskData->inputs_count[0]);
    cols_count = taskData->outputs_count[0] + 1;
    rows_count = matrix.size() / cols_count;
    res = std::vector<double>(taskData->outputs_count[0], std::numeric_limits<double>::quiet_NaN());
  } else {
    matrix = std::vector<double>();
    res = std::vector<double>();
  }

  return true;
}

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return world.rank() != 0 || (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] >= 0 &&
                               (taskData->inputs_count[0] % (taskData->outputs_count[0] + 1) == 0));
}

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, rows_count, 0);
  boost::mpi::broadcast(world, cols_count, 0);
  // std::cout<< world.rank() << " - " << rows_count << "x" << cols_count <<std::endl;

  std::vector<int> virtual_to_physical_row_idx(rows_count);
  std::vector<int> physical_to_virtual_row_idx(rows_count);
  if (world.rank() == 0) {
    for (int i = 0; i < rows_count; i++) {
      physical_to_virtual_row_idx[i] = virtual_to_physical_row_idx[i] = i;
    }
  }

  int avg_to_send = rows_count / world.size();
  int extra_to_send = rows_count % world.size();
  std::vector<int> sendcounts(world.size());
  std::vector<int> displs(world.size(), 0);
  for (int i = 0; i < world.size(); i++) {
    sendcounts[i] = (avg_to_send + (i < extra_to_send ? 1 : 0)) * cols_count;
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  std::vector<double> local_matrix(sendcounts[world.rank()]);
  std::vector<double> local_current_row(cols_count);
  int local_rows_count = local_matrix.size() / cols_count;

  auto get_abs_item = [&](int row, int col) { return matrix[virtual_to_physical_row_idx[row] * cols_count + col]; };

  // forward elimination
  for (int current_row = 0; current_row < rows_count; current_row++) {
    int pivot_row = 0;
    int pivot_col = 0;
    auto get_local_item = [&](int row, int col) -> auto & {
      return local_matrix[row * cols_count + col - displs[world.rank()]];
    };
    if (world.rank() == 0) {
      pivot_row = rows_count - 1;
      pivot_col = cols_count;
      for (int row = current_row; row < rows_count; row++) {
        for (int col = current_row; col < cols_count; col++) {
          if (get_abs_item(row, col) != 0) {
            if (col < pivot_col) {
              pivot_row = row;
              pivot_col = col;
            }
            break;
          }
        }
      }
      std::swap(physical_to_virtual_row_idx[virtual_to_physical_row_idx[current_row]],
                physical_to_virtual_row_idx[virtual_to_physical_row_idx[pivot_row]]);
      std::swap(virtual_to_physical_row_idx[current_row], virtual_to_physical_row_idx[pivot_row]);

      auto *current_row_start = &matrix[virtual_to_physical_row_idx[current_row] * cols_count];
      std::copy(current_row_start, current_row_start + cols_count, local_current_row.begin());
    }
    boost::mpi::broadcast(world, virtual_to_physical_row_idx.data(), virtual_to_physical_row_idx.size(), 0);
    boost::mpi::broadcast(world, physical_to_virtual_row_idx.data(), physical_to_virtual_row_idx.size(), 0);
    boost::mpi::broadcast(world, pivot_row, 0);
    boost::mpi::broadcast(world, pivot_col, 0);

    boost::mpi::broadcast(world, local_current_row.data(), cols_count, 0);
    boost::mpi::scatterv(world, matrix.data(), sendcounts, displs, local_matrix.data(), local_matrix.size(), 0);

    for (int row = displs[world.rank()] / cols_count; row < (displs[world.rank()] / cols_count) + local_rows_count;
         row++) {
      if (physical_to_virtual_row_idx[row] > current_row) {
        double ratio = get_local_item(row, pivot_col) / local_current_row[pivot_col];
        for (int col = pivot_col; col < cols_count; col++) {
          get_local_item(row, col) -= local_current_row[col] * ratio;
        }
      }
    }

    boost::mpi::gatherv(world, local_matrix.data(), local_matrix.size(), matrix.data(), sendcounts, displs, 0);
  }

  if (world.rank() == 0) {
    // back substitution
    for (int current_row = rows_count - 1; current_row >= 0; current_row--) {
      for (int col = 0; col < cols_count - 1; col++) {
        if (get_abs_item(current_row, col) != 0) {
          double known_part = 0;
          for (int variable_col = col + 1; variable_col < cols_count - 1; variable_col++) {
            if (get_abs_item(current_row, variable_col) != 0) {
              known_part += res[variable_col] * get_abs_item(current_row, variable_col);
            }
          }

          res[col] = -(get_abs_item(current_row, cols_count - 1) + known_part) / get_abs_item(current_row, col);

          break;
        }
      }
    }
  }
  return true;
}

bool rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  }
  return true;
}
