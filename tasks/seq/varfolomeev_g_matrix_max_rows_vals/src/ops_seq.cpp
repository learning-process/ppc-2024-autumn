// Copyright 2024 Nesterov Alexander
#include "seq/varfolomeev_g_matrix_max_rows_vals/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

int varfolomeev_g_matrix_max_rows_vals_seq::searchMaxInVec(std::vector<int> vec) {
    int max = vec[0];
    for (int i = 1; i < vec.size(); i++) {
      if (max < vec[i]) max = vec[i];
    }
    return max;
  }

std::vector<std::vector<int>> varfolomeev_g_matrix_max_rows_vals_seq::generateMatrix(const int rows, const int cols, int a, int b) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

  // set generator
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i][j] = std::rand() % (b - a + 1) + a;
    }
  }
  return matrix;
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_m = static_cast<size_t>(taskData->inputs_count[0]);    // rows count
  size_n = static_cast<size_t>(taskData->inputs_count[1]);    // columns count
  res_vec = std::vector<int>(size_m, 0);
  mtr.resize(size_m, std::vector<int>(size_n));
  
  for (int i = 0; i < size_m; i++) {
    auto* inpt_prt = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < size_n; j++) {
      mtr[i][j] = inpt_prt[j];
      //mtr[i][j] = inpt_prt[i*size_n + j];
    }
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 2 &&  // Checking that there are two elements in inputs_count
          taskData->inputs_count[0] >= 0 &&      // Checking that the number of rows is greater than 0
          taskData->inputs_count[1] >= 0 &&      // Checking that the number of columns is greater than 0
          taskData->outputs_count.size() == 1 && // Checking that there is one element in outputs_count
          taskData->outputs_count[0] == taskData->inputs_count[0]; // Checking that the number of output data is equal to the number of rows

  //return taskData->inputs_count[0] >=2 && taskData->inputs_count[1] >= 0 && static_cast<size_t>(taskData->outputs_count[0]) == size_m;
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::run() {
  internal_order_test();
  for (int i = 0; i < size_m; i++){
    res_vec[i] = searchMaxInVec(mtr[i]); 
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows::post_processing() {
  internal_order_test();
  for (int i = 0; i < size_m; i++){
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_vec[i];
  }
  return true;
}
