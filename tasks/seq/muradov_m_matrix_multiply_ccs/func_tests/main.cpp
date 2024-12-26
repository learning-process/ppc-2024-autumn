#include <gtest/gtest.h>

#include <random>

#include "seq/muradov_m_matrix_multiply_ccs/include/ops_seq.hpp"

namespace muradov_m_matrix_multiply_ccs_seq {

std::vector<std::vector<double>> gen_rand_matrix(int rows, int cols, int non_zero_count) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> row_dist(0, rows - 1);
  std::uniform_int_distribution<> col_dist(0, cols - 1);
  std::uniform_real_distribution<> value_dist(-10.0, 10.0);

  int count = 0;
  while (count < non_zero_count) {
    int r = row_dist(gen);
    int c = col_dist(gen);

    if (matrix[r][c] == 0.0) {
      matrix[r][c] = value_dist(gen);
      ++count;
    }
  }

  return matrix;
}

std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>> &A,
                                                   const std::vector<std::vector<double>> &B) {
  int rows_A = A.size();
  int cols_A = A[0].size();
  int cols_B = B[0].size();

  std::vector<std::vector<double>> result(rows_A, std::vector<double>(cols_B, 0.0));

  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      for (int k = 0; k < cols_A; ++k) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

void func_test_template(const std::vector<std::vector<double>> &A_, const std::vector<std::vector<double>> &B_) {
  std::vector<double> A_val;
  std::vector<int> A_row_ind;
  std::vector<int> A_col_ptr;
  int rows_A = A_.size();
  int cols_A = A_[0].size();

  std::vector<double> B_val;
  std::vector<int> B_row_ind;
  std::vector<int> B_col_ptr;
  int rows_B = B_.size();
  int cols_B = B_[0].size();

  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  auto exp_C = multiply_matrices(A_, B_);
  convert_to_CCS(exp_C, exp_C.size(), exp_C[0].size(), exp_C_val, exp_C_row_ind, exp_C_col_ptr);

  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  convert_to_CCS(A_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  convert_to_CCS(B_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_B));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_B));
  task_data->inputs_count.emplace_back(1);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
  task_data->inputs_count.emplace_back(A_val.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_ind.data()));
  task_data->inputs_count.emplace_back(A_row_ind.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data->inputs_count.emplace_back(A_col_ptr.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
  task_data->inputs_count.emplace_back(B_val.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_ind.data()));
  task_data->inputs_count.emplace_back(B_row_ind.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data->inputs_count.emplace_back(B_col_ptr.size());

  C_val.resize(exp_C_val.size());
  C_row_ind.resize(exp_C_row_ind.size());
  C_col_ptr.resize(exp_C_col_ptr.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_val.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_ind.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));

  MatrixMultiplyCCS task(task_data);
  if (task.validation()) {
    task.pre_processing();
    task.run();
    task.post_processing();

    ASSERT_EQ(exp_C_val, C_val);
    ASSERT_EQ(exp_C_row_ind, C_row_ind);
    ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
  }
}

void val_test_template(const std::vector<std::vector<double>> &A_, const std::vector<std::vector<double>> &B_) {
  int rows_A;
  int cols_A;
  if (A_.empty()) {
    rows_A = 0;
    cols_A = 0;
  } else {
    rows_A = A_.size();
    cols_A = A_[0].size();
  }

  std::vector<double> B_val;
  std::vector<int> B_row_ind;
  std::vector<int> B_col_ptr;
  int rows_B;
  int cols_B;
  if (B_.empty()) {
    rows_B = 0;
    cols_B = 0;
  } else {
    rows_B = B_.size();
    cols_B = B_[0].size();
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_B));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_B));
  task_data->inputs_count.emplace_back(1);

  MatrixMultiplyCCS task(task_data);
  EXPECT_FALSE(task.validation());
}

}  // namespace muradov_m_matrix_multiply_ccs_seq

TEST(muradov_m_matrix_multiply_ccs_seq, ValEmptyA) {
  std::vector<std::vector<double>> A_ = {};
  std::vector<std::vector<double>> B_ = {{0, 4, 0, 0, 1}, {5, 0, 0, 2, 0}, {0, 0, 3, 0, 6}};
  muradov_m_matrix_multiply_ccs_seq::val_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, ValEmptyB) {
  std::vector<std::vector<double>> A_ = {{1, 0, 2}, {0, 3, 0}};
  std::vector<std::vector<double>> B_ = {};
  muradov_m_matrix_multiply_ccs_seq::val_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, ValEmptyAB) {
  std::vector<std::vector<double>> A_ = {};
  std::vector<std::vector<double>> B_ = {};
  muradov_m_matrix_multiply_ccs_seq::val_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, SmallMatrices) {
  std::vector<std::vector<double>> A_ = {{1, 0, 2}, {0, 3, 0}};
  std::vector<std::vector<double>> B_ = {{0, 4, 0, 0, 1}, {5, 0, 0, 2, 0}, {0, 0, 3, 0, 6}};
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Random3x5And5x4) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(3, 5, 5);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(5, 4, 10);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Random10x5And5x10) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(10, 5, 5);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(5, 10, 10);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Random2x3And3x10) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(2, 3, 5);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(3, 10, 10);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Square4x4) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(4, 4, 10);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(4, 4, 10);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Rectangular6x2And2x6) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(6, 2, 5);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(2, 6, 5);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Vector1x5And5x1) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(1, 5, 5);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(5, 1, 5);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Identity3x3) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(3, 3, 5);
  std::vector<std::vector<double>> B_ = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Identity4x4) {
  std::vector<std::vector<double>> A_ = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(4, 4, 5);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}

TEST(muradov_m_matrix_multiply_ccs_seq, Large100x50And50x100) {
  auto A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(100, 50, 5);
  auto B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(50, 100, 5);
  muradov_m_matrix_multiply_ccs_seq::func_test_template(A_, B_);
}
