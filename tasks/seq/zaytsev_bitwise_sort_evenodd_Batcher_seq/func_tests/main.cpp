// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/zaytsev_bitwise_sort_evenodd_Batcher_seq/include/ops_seq.hpp"

TEST(zaytsev_bitwise_sort_evenodd_Batcher_seq, SingleElement) {
  const int count = 1;

  std::vector<int> in = {5};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential testTask(taskDataSeq);

  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], in[0]);
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher_seq, AlreadySorted) {
  const int count = 5;

  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(5, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential testTask(taskDataSeq);

  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out, in);
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher_seq, ReverseOrder) {
  const int count = 5;

  std::vector<int> in = {5, 4, 3, 2, 1};
  std::vector<int> out(5, 0);
  std::vector<int> expected = {1, 2, 3, 4, 5};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential testTask(taskDataSeq);

  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out, expected);
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher_seq, RandomOrder) {
  const int count = 6;

  std::vector<int> in = {10, -3, 7, 2, -8, 4};
  std::vector<int> out(6, 0);
  std::vector<int> expected = {-8, -3, 2, 4, 7, 10};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential testTask(taskDataSeq);

  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out, expected);
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher_seq, AllNegative) {
  const int count = 4;

  std::vector<int> in = {-10, -20, -30, -40};
  std::vector<int> out(4, 0);
  std::vector<int> expected = {-40, -30, -20, -10};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaytsev_bitwise_sort_evenodd_Batcher_seq::TestTaskSequential testTask(taskDataSeq);

  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out, expected);
}
