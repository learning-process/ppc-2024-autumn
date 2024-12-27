#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/savchenko_m_ribbon_mult_split_a/include/ops_seq_savchenko.hpp"

namespace savchenko_m_ribbon_mult_split_a_seq {
std::vector<int> getRandomMatrix(size_t rows, size_t columns, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  // Forming a random matrix
  std::vector<int> matrix(rows * columns);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      matrix[i * columns + j] = min + gen() % (max - min + 1);
    }
  }

  return matrix;
}

int getRandomInt(int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int rand_int = min + gen() % (max - min + 1);
  return rand_int;
}
}  // namespace savchenko_m_ribbon_mult_split_a_seq

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_zero_inputs) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_one_inputs) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_three_inputs) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_zero_outputs) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_two_outputs) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_zero_inputs_count) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_inputs_count_less_than_4) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_inputs_count_more_than_4) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_zero_outputs_count) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_outputs_count_more_than_1) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_matrix_A_zero_size) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->inputs_count.emplace_back(0);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_matrix_B_zero_size) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->inputs_count.emplace_back(0);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, validation_not_equal_columnsA_rowsB) {
  // Create data

  const int size = 5;
  const int columns_A = size;
  const int rows_A = size;
  const int columns_B = size;
  const int rows_B = size;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);

  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A - 1);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B + 1);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_ribbon_mult_split_a_seq, simple_matrixA_3x3_matrixB_3x3) {
  // Create data
  // const int gen_min = -1000;
  // const int gen_max = 1000;

  const int columns_A = 3;
  const int rows_A = 3;
  const int columns_B = 3;
  const int rows_B = 3;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);
  matrix_A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  matrix_B = {9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::vector<int> matrix_res(res_size, 0);
  std::vector<int> refference(res_size, 0);
  refference = {30, 24, 18, 84, 69, 54, 138, 114, 90};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, simple_matrixA_3x3_matrixB_3x2) {
  // Create data
  // const int gen_min = -1000;
  // const int gen_max = 1000;

  const int columns_A = 3;
  const int rows_A = 3;
  const int columns_B = 2;
  const int rows_B = 3;
  const int res_size = rows_A * columns_B;

  std::vector<int> matrix_A(rows_A * columns_A, 0);
  std::vector<int> matrix_B(rows_B * columns_B, 0);
  matrix_A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  matrix_B = {1, 2, 3, 4, 5, 6};

  std::vector<int> matrix_res(res_size, 0);
  std::vector<int> refference(res_size, 0);
  refference = {22, 28, 49, 64, 76, 100};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, matrix_10x10_and_10x10_with_rand_num) {
  // Create data
  const int columns_A = 10;
  const int rows_A = 10;
  const int columns_B = 10;
  const int rows_B = 10;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;
  const int num_A = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);
  const int num_B = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);

  std::vector<int> matrix_A(rows_A * columns_A, num_A);
  std::vector<int> matrix_B(rows_B * columns_B, num_B);
  std::vector<int> matrix_res(res_size, 0);

  const int num_ref = columns_A * num_A * num_B;
  std::vector<int> refference(res_size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, matrix_5x10_and_10x5_with_rand_num) {
  // Create data
  const int columns_A = 10;
  const int rows_A = 5;
  const int columns_B = 5;
  const int rows_B = 10;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;
  const int num_A = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);
  const int num_B = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);

  std::vector<int> matrix_A(rows_A * columns_A, num_A);
  std::vector<int> matrix_B(rows_B * columns_B, num_B);
  std::vector<int> matrix_res(res_size, 0);

  const int num_ref = columns_A * num_A * num_B;
  std::vector<int> refference(res_size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, matrix_5x10_and_10x1_with_rand_num) {
  // Create data
  const int columns_A = 10;
  const int rows_A = 5;
  const int columns_B = 1;
  const int rows_B = 10;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;
  const int num_A = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);
  const int num_B = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);

  std::vector<int> matrix_A(rows_A * columns_A, num_A);
  std::vector<int> matrix_B(rows_B * columns_B, num_B);
  std::vector<int> matrix_res(res_size, 0);

  const int num_ref = columns_A * num_A * num_B;
  std::vector<int> refference(res_size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, matrix_100x100_and_100x100_with_rand_num) {
  // Create data
  const int columns_A = 100;
  const int rows_A = 100;
  const int columns_B = 100;
  const int rows_B = 100;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;
  const int num_A = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);
  const int num_B = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);

  std::vector<int> matrix_A(rows_A * columns_A, num_A);
  std::vector<int> matrix_B(rows_B * columns_B, num_B);
  std::vector<int> matrix_res(res_size, 0);

  const int num_ref = columns_A * num_A * num_B;
  std::vector<int> refference(res_size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, matrix_50x100_and_100x50_with_rand_num) {
  // Create data
  const int columns_A = 100;
  const int rows_A = 50;
  const int columns_B = 50;
  const int rows_B = 100;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;
  const int num_A = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);
  const int num_B = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);

  std::vector<int> matrix_A(rows_A * columns_A, num_A);
  std::vector<int> matrix_B(rows_B * columns_B, num_B);
  std::vector<int> matrix_res(res_size, 0);

  const int num_ref = columns_A * num_A * num_B;
  std::vector<int> refference(res_size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}

TEST(savchenko_m_ribbon_mult_split_a_seq, matrix_50x100_and_100x1_with_rand_num) {
  // Create data
  const int columns_A = 100;
  const int rows_A = 50;
  const int columns_B = 1;
  const int rows_B = 100;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;
  const int num_A = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);
  const int num_B = savchenko_m_ribbon_mult_split_a_seq::getRandomInt(gen_min, gen_max);

  std::vector<int> matrix_A(rows_A * columns_A, num_A);
  std::vector<int> matrix_B(rows_B * columns_B, num_B);
  std::vector<int> matrix_res(res_size, 0);

  const int num_ref = columns_A * num_A * num_B;
  std::vector<int> refference(res_size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(refference, matrix_res);
}
