// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/sorokin_a_graham_algorithm/include/ops_seq.hpp"

TEST(sorokin_a_graham_algorithm_seq, Base) {
  // Create data
  std::vector<int> in = {0, 0, 4, 0, 4, 4, 0, 4, 2, 2, 1, 3};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {0, 0, 4, 0, 4, 4, 0, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sorokin_a_graham_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < outres.size(); i++) ASSERT_EQ(outres[i], out[i]);
}
TEST(sorokin_a_graham_algorithm_seq, Base1) {
  // Create data
  std::vector<int> in = {12, 5, 2, 3,  5, 2, 7, 1, 8,  6, 3, 8, 6, 5, 9, 3, 4, 7, 10, 10,
                         2,  9, 5, 12, 0, 0, 8, 0, 10, 7, 6, 9, 1, 5, 3, 2, 2, 8, 7,  10};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {0, 0, 8, 0, 12, 5, 10, 10, 5, 12, 2, 9, 1, 5};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sorokin_a_graham_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < outres.size(); i++) ASSERT_EQ(outres[i], out[i]);
}
TEST(sorokin_a_graham_algorithm_seq, Base2) {
  // Create data
  std::vector<int> in = {1, 4, 2, 8, 6, 4, 9, 3, 7, 6, 2, 2, 5, 1, 4, 9, 10, 10, 8, 2};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {5, 1, 8, 2, 9, 3, 10, 10, 4, 9, 2, 8, 1, 4, 2, 2};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sorokin_a_graham_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < outres.size(); i++) ASSERT_EQ(outres[i], out[i]);
}
TEST(sorokin_a_graham_algorithm_seq, Base3) {
  // Create data
  std::vector<int> in = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {1, 1, 5, 5};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sorokin_a_graham_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < outres.size(); i++) ASSERT_EQ(outres[i], out[i]);
}
TEST(sorokin_a_graham_algorithm_seq, Base4) {
  // Create data
  std::vector<int> in = {1, 1, 2, 2, 3, 1, 2, 3, 0, 0, 3, 3};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {0, 0, 3, 1, 3, 3, 2, 3};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sorokin_a_graham_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < outres.size(); i++) ASSERT_EQ(outres[i], out[i]);
}
TEST(sorokin_a_graham_algorithm_seq, Base5) {
  // Create data
  std::vector<int> in = {-1, 0, 0, -1, -1, -1, -2, -1, -1, -2, -1, 0};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {-1, -2, 0, -1, -1, 0, -2, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sorokin_a_graham_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < outres.size(); i++) ASSERT_EQ(outres[i], out[i]);
}
