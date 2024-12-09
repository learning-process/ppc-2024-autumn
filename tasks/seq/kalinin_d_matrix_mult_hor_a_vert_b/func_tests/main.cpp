#include <gtest/gtest.h>

#include <vector>

#include "seq/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_seq.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_seq {
void get_random_matrix(std::vector<int>& mat) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-500, 499);
  for (auto& elem : mat) {
    elem = dist(gen);
  }
}

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_seq

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, square_matrix_4x4) {
  std::vector<size_t> dimensions = {3, 2, 2, 3};
  std::vector<int> matrix_a = {1, 2, 3, 4, 5, 6};
  std::vector<int> matrix_b = {7, 8, 9, 10, 11, 12};
  std::vector<int> expected_result = {27, 30, 33, 61, 68, 75, 95, 106, 117};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, out_1x1) {
  std::vector<size_t> dimensions = {1, 1, 1, 1};
  std::vector<int> matrix_a = {5};
  std::vector<int> matrix_b = {7};
  std::vector<int> expected_result = {35};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, out_4x4) {
  std::vector<size_t> dimensions = {4, 4, 4, 4};
  std::vector<int> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> matrix_b = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> expected_result = {80, 70, 60, 50, 240, 214, 188, 162, 400, 358, 316, 274, 560, 502, 444, 386};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, large_numbers) {
  std::vector<size_t> dimensions = {2, 2, 2, 2};
  std::vector<int> matrix_a = {1000, 2000, 3000, 4000};
  std::vector<int> matrix_b = {5000, 6000, 7000, 8000};
  std::vector<int> expected_result = {19000000, 22000000, 43000000, 50000000};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, same_elements) {
  std::vector<size_t> dimensions = {2, 3, 3, 2};
  std::vector<int> matrix_a = {1, 1, 1, 1, 1, 1};
  std::vector<int> matrix_b = {1, 1, 1, 1, 1, 1};
  std::vector<int> expected_result = {3, 3, 3, 3};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, in_3x3) {
  std::vector<size_t> dimensions = {3, 3, 3, 3};
  std::vector<int> matrix_a = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<int> matrix_b = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<int> expected_result = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, matrix_0) {
  std::vector<size_t> dimensions = {3, 3, 3, 3};
  std::vector<int> matrix_a = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> matrix_b = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> expected_result = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, negative_elements) {
  std::vector<size_t> dimensions = {2, 2, 2, 2};
  std::vector<int> matrix_a = {-1, -2, -3, -4};
  std::vector<int> matrix_b = {1, 2, 3, 4};
  std::vector<int> expected_result = {-7, -10, -15, -22};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, in_2x3_3x2) {
  std::vector<size_t> dimensions = {2, 3, 3, 2};
  std::vector<int> matrix_a = {1, 2, 3, 4, 5, 6};
  std::vector<int> matrix_b = {7, 8, 9, 10, 11, 12};
  std::vector<int> expected_result = {58, 64, 139, 154};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, in_2x2) {
  std::vector<size_t> dimensions = {2, 2, 2, 2};
  std::vector<int> matrix_a = {1, 2, 3, 4};
  std::vector<int> matrix_b = {5, 6, 7, 8};
  std::vector<int> expected_result = {19, 22, 43, 50};

  const size_t row_a = dimensions[0];
  const size_t col_b = dimensions[3];

  std::vector<int> matrix_c(row_a * col_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  ASSERT_EQ(matrix_c, expected_result);
}
