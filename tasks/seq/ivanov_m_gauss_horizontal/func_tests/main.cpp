// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/ivanov_m_gauss_horizontal/include/ops_seq.hpp"

namespace ivanov_m_gauss_horizontal_seq {
std::vector<double> GenSolution(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> solution(size, 0);

  for (int i = 0; i < size; i++) {
    solution[i] = static_cast<double>(generator(gen));  // generating random coefficient in range [-2, 2]
  }
  return solution;
}

std::vector<double> GenMatrix(const std::vector<double> &solution) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);

  std::vector<double> extended_matrix;
  int size = static_cast<int>(solution.size());

  // generate identity matrix
  for (int row = 0; row < size; row++) {
    for (int column = 0; column < size; column++) {
      if (row == column) {
        extended_matrix.push_back(1);
      } else {
        extended_matrix.push_back(0);
      }
    }
    extended_matrix.push_back(solution[row]);
  }

  // saturation left triangle
  for (int row = 1; row < size; row++) {
    for (int column = 0; column < row; column++) {
      extended_matrix[get_linear_index(row, column, size + 1)] +=
          extended_matrix[get_linear_index(row - 1, column, size + 1)];
    }
    extended_matrix[get_linear_index(row, size, size + 1)] +=
        extended_matrix[get_linear_index(row - 1, size, size + 1)];
  }

  // saturation of matrix by random numbers
  for (int row = size - 1; row > 0; row--) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row - 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  // saturation of matrix by random numbers
  for (int row = 0; row < size - 1; row++) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row + 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  return extended_matrix;
}
}  // namespace ivanov_m_gauss_horizontal_seq

TEST(ivanov_m_gauss_horizontal_seq_func_test, validation_false_test_inputs) {
  int n = 2;
  std::vector<double> matrix;
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  EXPECT_EQ(testTaskSequential.validation(), false);
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, validation_false_test_size_of_matrix) {
  int n = 3;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  EXPECT_EQ(testTaskSequential.validation(), false);
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, pre_processing_false_test_determinant_is_zero_because_rows_are_zero) {
  int n = 2;
  std::vector<double> matrix = {0, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  EXPECT_EQ(testTaskSequential.pre_processing(), false);
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, pre_processing_false_test_determinant_is_counted_and_equals_zero) {
  int n = 2;
  std::vector<double> matrix = {1, 1, 1, 1, 1, 1};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  EXPECT_EQ(testTaskSequential.pre_processing(), false);
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, pre_processing_true_test_determinant_is_not_zero) {
  int n = 2;
  std::vector<double> matrix = {2, 0, 1, 0, 4, 1};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  EXPECT_EQ(testTaskSequential.pre_processing(), true);
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, run_true_matrix_size_2) {
  int n = 2;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out[i], ans[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, run_true_matrix_size_3_simple) {
  int n = 3;
  std::vector<double> matrix = {0, 0, 1, 3, 0, 1, 0, 2, 1, 0, 0, 4};
  std::vector<double> ans = {4, 2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out[i], ans[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, run_true_random_matrix_size_5) {
  int n = 5;
  std::vector<double> ans = ivanov_m_gauss_horizontal_seq::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_seq::GenMatrix(ans);
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out[i], ans[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_seq_func_test, run_true_random_matrix_size_7) {
  int n = 7;
  std::vector<double> ans = ivanov_m_gauss_horizontal_seq::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_seq::GenMatrix(ans);
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_gauss_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out[i], ans[i], 1e-3);
  }
}