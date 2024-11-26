// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/opolin_d_max_of_matrix_elements/include/ops_seq.hpp"

std::vector<int> getRandomVectorForGetMaxInMatrix(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrixForGetMaxInMatrix(int rows, int cols, int min, int max) {
  std::vector<std::vector<int>> matr(rows);
  for (int i = 0; i < rows; i++) {
    matr[i] = getRandomVectorForGetMaxInMatrix(cols, min, max);
  }
  return matr;
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_1000x1000) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int rows = 1000;
  const int cols = 1000;
  const int max = 1000;
  const int min = 0;
  int res = std::numeric_limits<int32_t>::max();

  std::vector<int> out(1, std::numeric_limits<int32_t>::min());
  std::vector<std::vector<int>> in = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
  int rand_c = gen() % cols;
  int rand_r = gen() % rows;
  in[rand_r][rand_c] = res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_100x1) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int rows = 100;
  const int cols = 1;
  const int max = 1000;
  const int min = 0;
  int res = std::numeric_limits<int32_t>::max();

  std::vector<int> out(1, std::numeric_limits<int32_t>::min());
  std::vector<std::vector<int>> in = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
  int rand_r = gen() % rows;
  in[rand_r][0] = res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_1x100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int rows = 1;
  const int cols = 100;
  const int max = 1000;
  const int min = 0;
  int res = std::numeric_limits<int32_t>::max();

  std::vector<int> out(1, std::numeric_limits<int32_t>::min());
  std::vector<std::vector<int>> in = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
  int rand_c = gen() % cols;
  in[0][rand_c] = res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Empty_Matrix) {
  const int rows = 0;
  const int cols = 0;
  std::vector<int> out(1, std::numeric_limits<int32_t>::min());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_Negative) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int rows = 100;
  const int cols = 100;
  const int max = -10;
  const int min = -1000;
  int expected = -5;

  std::vector<int> out(1, std::numeric_limits<int32_t>::min());
  std::vector<std::vector<int>> in = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
  int rand_c = gen() % cols;
  int rand_r = gen() % rows;
  in[rand_r][rand_c] = expected;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Matrix_Negative_Size) {
  const int rows = -10;
  const int cols = -10;
  std::vector<int> out(1, std::numeric_limits<int32_t>::min());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), out[0]);
}