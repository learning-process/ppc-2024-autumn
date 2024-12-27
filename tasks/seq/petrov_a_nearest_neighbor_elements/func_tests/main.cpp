// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/petrov_a_nearest_neighbor_elements/include/ops_seq.hpp"

TEST(petrov_a_nearest_neighbor_elements_seq, SUM20) {
  // Create data
  std::vector<int> in{8, 3};
  std::vector<int> out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_a_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<int> T{8, 3};
  ASSERT_EQ(T[0], out[0]);
  ASSERT_EQ(T[1], out[1]);
}

TEST(petrov_a_nearest_neighbor_elements_seq, SUM50) {
  // Create data
  std::vector<int> in{-10, -5, -3, 2, 7, 12};
  std::vector<int> out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_a_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();

  testTaskSequential.post_processing();

  std::vector<int> T{-5, -3};
  ASSERT_EQ(T[0], out[0]);
  ASSERT_EQ(T[1], out[1]);
}

TEST(petrov_a_nearest_neighbor_elements_seq, SUM70) {
  // Create data
  std::vector<int> in{10, 8, 6, 4, 2, 0};
  std::vector<int> out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_a_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<int> T{10, 8};
  ASSERT_EQ(T[0], out[0]);
  ASSERT_EQ(T[1], out[1]);
}

TEST(petrov_a_nearest_neighbor_elements_seq, SUM100) {
  // Create data
  std::vector<int> in{5, 5, 5, 5, 5, 5};
  std::vector<int> out(2, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  petrov_a_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<int> T{5, 5};
  ASSERT_EQ(T[0], out[0]);
  ASSERT_EQ(T[1], out[1]);
}