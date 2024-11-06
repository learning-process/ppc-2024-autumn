#include "seq/varfolomeev_g_matrix_max_rows_vals/include/ops_seq.hpp"

#include <cstddef>
#include <thread>

using namespace std::chrono_literals;

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_m = taskData->inputs_count[0];  // rows count
  size_n = taskData->inputs_count[1];  // columns count
  res_vec = std::vector<int>(size_m, 0);
  mtr.resize(size_m, std::vector<int>(size_n));

  for (int i = 0; i < size_m; i++) {
    auto* inpt_prt = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < size_n; j++) {
      mtr[i][j] = inpt_prt[j];
      // mtr[i][j] = inpt_prt[i*size_n + j];
    }
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 2 &&   // Checking that there are two elements in inputs_count
         taskData->inputs_count[0] >= 0 &&       // Checking that the number of rows is greater than 0
         taskData->inputs_count[1] >= 0 &&       // Checking that the number of columns is greater than 0
         taskData->outputs_count.size() == 1 &&  // Checking that there is one element in outputs_count
         taskData->outputs_count[0] ==
             taskData->inputs_count[0];  // Checking that the number of output data is equal to the number of rows
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::run() {
  internal_order_test();
  int maxInRow = mtr[0][0];
  for (int i = 0; i < size_m; i++) {
    maxInRow = mtr[i][0];
    for (int j = 0; j < size_n; j++) {
      if (maxInRow < mtr[i][j]) maxInRow = mtr[i][j];
    }
    res_vec[i] = maxInRow;
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::post_processing() {
  internal_order_test();
  for (int i = 0; i < size_m; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_vec[i];
  }
  return true;
}
