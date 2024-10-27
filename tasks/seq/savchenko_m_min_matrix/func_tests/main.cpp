// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <climits>

#include "seq/savchenko_m_min_matrix/include/ops_seq.hpp"

TEST(savchenko_m_min_matrix_seq, test_min_10x10) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 10;
  const int columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;
  int ref = INT_MIN;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<int> in = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);

  int index = gen() % (rows * columns);
  in[index] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(savchenko_m_min_matrix_seq, test_min_100x10) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 100;
  const int columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;
  int ref = INT_MIN;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<int> in = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);

  int index = gen() % (rows * columns);
  in[index] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(savchenko_m_min_matrix_seq, test_min_10x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 10;
  const int columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  int ref = INT_MIN;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<int> in = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);

  int index = gen() % (rows * columns);
  in[index] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(savchenko_m_min_matrix_seq, test_min_100x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 100;
  const int columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  int ref = INT_MIN;

  // Create data
  std::vector<int> out(1, INT_MAX);
  std::vector<int> in = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);

  int index = gen() % (rows * columns);
  in[index] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
