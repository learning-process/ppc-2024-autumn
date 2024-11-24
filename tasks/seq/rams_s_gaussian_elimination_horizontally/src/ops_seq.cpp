#include "seq/rams_s_gaussian_elimination_horizontally/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

bool rams_s_gaussian_elimination_horizontally_seq::TaskSequential::pre_processing() {
  internal_order_test();

  auto *input_data = reinterpret_cast<double *>(taskData->inputs[0]);
  matrix = std::vector<double>(input_data, input_data + taskData->inputs_count[0]);
  cols_count = taskData->outputs_count[0] + 1;
  rows_count = matrix.size() / cols_count;
  res = std::vector<double>(taskData->outputs_count[0], std::numeric_limits<double>::quiet_NaN());
  return true;
}

bool rams_s_gaussian_elimination_horizontally_seq::TaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] >= 0 &&
         (taskData->inputs_count[0] % (taskData->outputs_count[0] + 1) == 0);
}

bool rams_s_gaussian_elimination_horizontally_seq::TaskSequential::run() {
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
        *get_item_ptr(row, col) -= get_item(current_row, col) * ratio;
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

bool rams_s_gaussian_elimination_horizontally_seq::TaskSequential::post_processing() {
  internal_order_test();

  std::copy(res.begin(), res.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}
