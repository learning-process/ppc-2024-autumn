#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shvedova_v_matrix_mult_horizontal_a_vertical_b_seq/include/ops_seq.hpp"

void RunMatrixMultiplicationTest(const std::vector<size_t>& in, const std::vector<int>& matrix_a,
                                 const std::vector<int>& matrix_b, const std::vector<int>& expected_result) {
  const size_t row_a = in[0];
  const size_t col_b = in[3];

  std::vector<int> matrix_c(row_a * col_b);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<size_t*>(in.data())));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_a.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (size_t i = 0; i < matrix_c.size(); ++i) {
    ASSERT_EQ(matrix_c[i], expected_result[i]);
  }
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_1x1_1x1) {
  std::vector<size_t> in = {1, 1, 1, 1};
  std::vector<int> matrix_a = {5};
  std::vector<int> matrix_b = {6};
  std::vector<int> expected_result = {30};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_2x3_3x1) {
  std::vector<size_t> in = {2, 3, 3, 1};
  std::vector<int> matrix_a = {1, -2, 3, 4, -5, 6};
  std::vector<int> matrix_b = {1, 2, 3};
  std::vector<int> expected_result = {6, 12};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_3x2_2x3) {
  std::vector<size_t> in = {3, 2, 2, 3};
  std::vector<int> matrix_a = {1, 2, 3, 4, 5, 6};
  std::vector<int> matrix_b = {7, 8, 9, 10, 11, 12};
  std::vector<int> expected_result = {27, 30, 33, 61, 68, 75, 95, 106, 117};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_2x4_4x2) {
  std::vector<size_t> in = {2, 4, 4, 2};
  std::vector<int> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> matrix_b = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expected_result = {50, 60, 114, 140};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_3x3_3x3) {
  std::vector<size_t> in = {3, 3, 3, 3};
  std::vector<int> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> matrix_b = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> expected_result = {30, 24, 18, 84, 69, 54, 138, 114, 90};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_1x3_3x1) {
  std::vector<size_t> in = {1, 3, 3, 1};
  std::vector<int> matrix_a = {3, 4, 2};
  std::vector<int> matrix_b = {1, 2, 3};
  std::vector<int> expected_result = {17};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_matrix_4x4_4x4) {
  std::vector<size_t> in = {4, 4, 4, 4};
  std::vector<int> matrix_a = {1, 0, 2, -1, 3, 1, 0, 2, 1, 2, 1, 1, 0, -1, 2, 1};
  std::vector<int> matrix_b = {2, -1, 0, 1, 1, 0, 3, -1, 0, 1, 2, 0, 1, 3, -1, 2};
  std::vector<int> expected_result = {1, -2, 5, -1, 9, 3, 1, 6, 5, 3, 7, 1, 0, 5, 0, 3};

  RunMatrixMultiplicationTest(in, matrix_a, matrix_b, expected_result);
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, validation_fails_on_mismatched_dimensions) {
  std::vector<size_t> in = {4, 4, 3, 3};
  std::vector<int> matrix_a = {1, 0, 2, -1, 3, 1, 0, 2, 1, 2, 1, 1, 0, -1, 2, 1};
  std::vector<int> matrix_b = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> expected_result = {1, -2, 5, -1, 9, 3, 1, 6, 5, 3, 7, 1, 0, 5, 0, 3};

  const size_t row_a = in[0];
  const size_t col_b = in[3];

  std::vector<int> matrix_c(row_a * col_b);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<size_t*>(in.data())));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_a.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential matrixTask(taskDataSeq);

  ASSERT_FALSE(matrixTask.validation());
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, validation_fails_on_incorrect_output) {
  std::vector<size_t> in = {4, 4, 4, 4};
  std::vector<int> matrix_a = {1, 0, 2, -1, 3, 1, 0, 2, 1, 2, 1, 1, 0, -1, 2, 1};
  std::vector<int> matrix_b = {2, -1, 0, 1, 1, 0, 3, -1, 0, 1, 2, 0, 1, 3, -1, 2};
  std::vector<int> expected_result = {1, -2, 5, -1, 9, 3, 1, 6, 5, 3, 7, 1, 0, 5, 0, 3};

  const size_t row_a = in[0];
  const size_t col_b = in[3];

  std::vector<int> matrix_c(row_a * col_b);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<size_t*>(in.data())));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_a.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size() + 1);

  shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential matrixTask(taskDataSeq);

  ASSERT_FALSE(matrixTask.validation());
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, validation_fails_on_zero_size) {
  std::vector<size_t> in = {4, 0, 4, 4};
  std::vector<int> matrix_a = {1, 0, 2, -1, 3, 1, 0, 2, 1, 2, 1, 1, 0, -1, 2, 1};
  std::vector<int> matrix_b = {2, -1, 0, 1, 1, 0, 3, -1, 0, 1, 2, 0, 1, 3, -1, 2};
  std::vector<int> expected_result = {1, -2, 5, -1, 9, 3, 1, 6, 5, 3, 7, 1, 0, 5, 0, 3};

  const size_t row_a = in[0];
  const size_t col_b = in[3];

  std::vector<int> matrix_c(row_a * col_b);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<size_t*>(in.data())));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_a.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->inputs_count.emplace_back(matrix_b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size() + 1);

  shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential matrixTask(taskDataSeq);

  ASSERT_FALSE(matrixTask.validation());
}