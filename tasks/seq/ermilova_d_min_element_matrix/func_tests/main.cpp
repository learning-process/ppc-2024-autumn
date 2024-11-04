// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>

#include "seq/ermilova_d_min_element_matrix/include/ops_seq.hpp"

std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrix(int rows, int cols, int upper_border, int lower_border) {
  if (rows <= 0 || cols <= 0) throw "Incorrect size";
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(cols, upper_border, lower_border);
  }
  return vec;
}

TEST(ermilova_d_min_element_matrix_seq, Can_create_vector) {
  const int size_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_seq, Cant_create_incorrect_vector) {
  const int size_test = -10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_seq, Can_create_matrix) {
  const int rows_test = 10;
  const int cols_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_seq, Cant_create_incorrect_matrix) {
  const int rows_test = -10;
  const int cols_test = 0;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_seq, Test_min_matrix_1x1) {
  const int rows_test = 1;
  const int cols_test = 1;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  int reference_min = -5000;

  // Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_min_element_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference_min, out[0]);
}

TEST(ermilova_d_min_element_matrix_seq, Test_min_matrix_10x10) {
  const int rows_test = 10;
  const int cols_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  int reference_min = -500;

  // Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_min_element_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference_min, out[0]);
}

TEST(ermilova_d_min_element_matrix_seq, Test_min_matrix_100x100) {
  const int rows_test = 100;
  const int cols_test = 100;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  int reference_min = -5000;

  // Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_min_element_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference_min, out[0]);
}

TEST(ermilova_d_min_element_matrix_seq, Test_min_matrix_50x100) {
  const int rows_test = 50;
  const int cols_test = 100;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  int reference_min = -5000;

  // Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_min_element_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference_min, out[0]);
}

TEST(ermilova_d_min_element_matrix_seq, Test_min_matrix_100x50) {
  const int rows_test = 100;
  const int cols_test = 50;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  int reference_min = -5000;

  // Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_min_element_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference_min, out[0]);
}
