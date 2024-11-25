#include "seq/shulpin_i_strip_hA_vB/include/strip_hA_vB.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

std::vector<int> shulpin_strip_scheme_A_B::get_RND_matrix(int col, int row) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 99);
  std::vector<int> rnd_matrix(col * row);
  int i, j;
  for (i = 0; i < row; ++i) {
    for (j = 0; j < col; ++j) {
      rnd_matrix[i * col + j] = dist(gen);
    }
  }
  return rnd_matrix;
}

void shulpin_strip_scheme_A_B::calculate_seq(int rows_a, int cols_a, int cols_b, std::vector<int> A_seq,
                                             std::vector<int> B_seq, std::vector<int>& C_seq) {
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      for (int k = 0; k < cols_a; ++k) {
        C_seq[i * cols_b + j] += A_seq[i * cols_a + k] * B_seq[k * cols_b + j];
      }
    }
  }
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::pre_processing() {
  internal_order_test();

  int cols_A_tmp = *reinterpret_cast<int*>(taskData->inputs[2]);
  int rows_A_tmp = *reinterpret_cast<int*>(taskData->inputs[3]);

  seq_cols_A = cols_A_tmp;
  seq_rows_A = rows_A_tmp;

  int cols_B_tmp = *reinterpret_cast<int*>(taskData->inputs[4]);
  int rows_B_tmp = *reinterpret_cast<int*>(taskData->inputs[5]);

  seq_cols_B = cols_B_tmp;
  seq_rows_B = rows_B_tmp;

  std::vector<int> A_tmp{};
  std::vector<int> B_tmp{};

  int* A_tmp_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int A_tmp_size = taskData->inputs_count[0];
  A_tmp.assign(A_tmp_data, A_tmp_data + A_tmp_size);

  int* B_tmp_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int B_tmp_size = taskData->inputs_count[1];
  B_tmp.assign(B_tmp_data, B_tmp_data + B_tmp_size);

  seq_A = A_tmp;
  seq_B = B_tmp;

  int res_size = taskData->outputs_count[0];
  seq_result.resize(res_size, 0);

  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::validation() {
  internal_order_test();

  int a_cols = *reinterpret_cast<int*>(taskData->inputs[2]);
  int a_rows = *reinterpret_cast<int*>(taskData->inputs[3]);
  int b_cols = *reinterpret_cast<int*>(taskData->inputs[4]);
  int b_rows = *reinterpret_cast<int*>(taskData->inputs[5]);

  return (taskData->inputs_count.size() > 4 && !taskData->outputs_count.empty() &&
          (a_cols > 0 && a_rows > 0 && b_cols > 0 && b_rows > 0) && (a_cols == b_rows));
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::run() {
  internal_order_test();

  calculate_seq(seq_rows_A, seq_cols_A, seq_cols_B, seq_A, seq_B, seq_result);

  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::post_processing() {
  internal_order_test();

  int* output = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(seq_result.begin(), seq_result.end(), output);

  return true;
}