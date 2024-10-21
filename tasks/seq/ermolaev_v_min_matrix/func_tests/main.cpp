// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/ermolaev_v_min_matrix/include/ops_seq.hpp"

TEST(ermolaev_v_min_matrix_seq, test_min_10x10) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 10;
  const int count_columns = 10;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = ermolaev_v_min_matrix_seq::getRandomMatrix(count_rows, count_columns);

  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = -1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermolaev_v_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(-1, out[0]);
}

TEST(ermolaev_v_min_matrix_seq, test_min_10x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 10;
  const int count_columns = 50;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = ermolaev_v_min_matrix_seq::getRandomMatrix(count_rows, count_columns);
  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = -1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermolaev_v_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(-1, out[0]);
}

TEST(ermolaev_v_min_matrix_seq, test_min_100x10) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 100;
  const int count_columns = 10;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = ermolaev_v_min_matrix_seq::getRandomMatrix(count_rows, count_columns);

  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = -1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermolaev_v_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(-1, out[0]);
}

TEST(ermolaev_v_min_matrix_seq, test_min_100x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 100;
  const int count_columns = 100;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = ermolaev_v_min_matrix_seq::getRandomMatrix(count_rows, count_columns);

  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = -1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ermolaev_v_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(-1, out[0]);
}
