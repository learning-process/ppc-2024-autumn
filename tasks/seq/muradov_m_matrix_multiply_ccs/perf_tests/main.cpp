#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/muradov_m_matrix_multiply_ccs/include/ops_seq.hpp"

namespace muradov_m_matrix_multiply_ccs_seq {

std::vector<std::vector<double>> gen_rand_matrix(int rows, int cols, int non_zero_count) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> row_dist(0, rows - 1);
  std::uniform_int_distribution<> col_dist(0, cols - 1);
  std::uniform_real_distribution<> value_dist(-128.0, 128.0);

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

}  // namespace muradov_m_matrix_multiply_ccs_seq

TEST(muradov_m_matrix_multiply_ccs, test_pipeline_run) {
  int size = 256;
  int elements = 6553;
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;

  A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(size, size, elements);
  B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(size, size, elements);

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

  auto exp_C = muradov_m_matrix_multiply_ccs_seq::multiply_matrices(A_, B_);
  muradov_m_matrix_multiply_ccs_seq::convert_to_CCS(exp_C, exp_C.size(), exp_C[0].size(), exp_C_val, exp_C_row_ind,
                                                    exp_C_col_ptr);

  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  muradov_m_matrix_multiply_ccs_seq::convert_to_CCS(A_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  muradov_m_matrix_multiply_ccs_seq::convert_to_CCS(B_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);

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
  task_data->outputs_count.emplace_back(C_val.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_ind.data()));
  task_data->outputs_count.emplace_back(C_row_ind.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data->outputs_count.emplace_back(C_col_ptr.size());

  auto task = std::make_shared<muradov_m_matrix_multiply_ccs_seq::MatrixMultiplyCCS>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(exp_C_val, C_val);
  ASSERT_EQ(exp_C_row_ind, C_row_ind);
  ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
}

TEST(sequential_muradov_m_matrix_multiply_ccs_perf_test, test_task_run) {
  int size = 256;
  int elements = 6553;
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;

  A_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(size, size, elements);
  B_ = muradov_m_matrix_multiply_ccs_seq::gen_rand_matrix(size, size, elements);

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

  auto exp_C = muradov_m_matrix_multiply_ccs_seq::multiply_matrices(A_, B_);
  muradov_m_matrix_multiply_ccs_seq::convert_to_CCS(exp_C, exp_C.size(), exp_C[0].size(), exp_C_val, exp_C_row_ind,
                                                    exp_C_col_ptr);

  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  muradov_m_matrix_multiply_ccs_seq::convert_to_CCS(A_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  muradov_m_matrix_multiply_ccs_seq::convert_to_CCS(B_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);

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
  task_data->outputs_count.emplace_back(C_val.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_ind.data()));
  task_data->outputs_count.emplace_back(C_row_ind.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data->outputs_count.emplace_back(C_col_ptr.size());

  auto task = std::make_shared<muradov_m_matrix_multiply_ccs_seq::MatrixMultiplyCCS>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(exp_C_val, C_val);
  ASSERT_EQ(exp_C_row_ind, C_row_ind);
  ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
}
