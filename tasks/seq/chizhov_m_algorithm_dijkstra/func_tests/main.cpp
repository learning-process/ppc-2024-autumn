// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/chizhov_m_algorithm_dijkstra/include/ops_seq.hpp"

TEST(chizhov_m_dijkstra_realization, Test_Graph_3x3) {
  int size = 3;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 2, 5, 4, 0, 2, 3, 1, 0};
  std::vector<int> res(size, 0);
  std::vector<int> ans = {0, 2, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  chizhov_m_dijkstra_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, res);
}

TEST(chizhov_m_dijkstra_realization_seq, Test_Graph_4x4) {
  int size = 4;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 9, 9, 3, 6, 0, 3, 5, 1, 3, 0, 5, 2, 2, 10, 0};
  std::vector<int> res(size, 0);
  std::vector<int> ans = {0, 5, 8, 3};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  chizhov_m_dijkstra_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, res);
}

TEST(chizhov_m_dijkstra_realization_seq, Test_Graph_5x5) {
  int size = 5;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 5, 0, 3, 0, 0, 0, 4, 2, 2, 0, 0, 0, 3, 0, 0, 3, 0, 0, 2, 9, 0, 1, 0, 0};
  std::vector<int> res(size, 0);
  std::vector<int> ans = {0, 5, 6, 3, 5};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  chizhov_m_dijkstra_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, res);
}

TEST(chizhov_m_dijkstra_realization_seq, Test_Negative_Value) {
  int size = 3;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 2, 5, -4, 0, 2, 3, 1, 0};
  std::vector<int> res(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  chizhov_m_dijkstra_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(chizhov_m_dijkstra_realization_seq, Test_Source_Vertex_False) {
  int size = 3;
  int st = 5;
  // Create data
  std::vector<int> matrix = {0, 2, 5, 4, 0, 2, 3, 1, 0};
  std::vector<int> res(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  chizhov_m_dijkstra_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}