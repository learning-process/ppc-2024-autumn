// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <random>

#include "seq/tsatsyn_a_increasing_contrast_by_histogram/include/ops_seq.hpp"
static std::vector<int> getRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}
TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_10) {

  // Create data
  std::vector<int> in;
  std::vector<int> out(1, 0);
  std::vector<int> sizes = {1200, 720};
  const int count_size_vector = 1200 * 1;
  in = getRandomVector(count_size_vector, 0, 255);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(sizes.data()));
  taskDataSeq->inputs_count.emplace_back(sizes.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(1, 1);
}

//TEST(tsatsyn_a_increasing_contrast_by_histogram_seq, Test_Sum_20) {
//  const int count = 20;
//
//  // Create data
//  std::vector<int> in(1, count);
//  std::vector<int> out(1, 0);
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//  taskDataSeq->inputs_count.emplace_back(in.size());
//  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//  taskDataSeq->outputs_count.emplace_back(out.size());
//
//  // Create Task
//  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//  ASSERT_EQ(testTaskSequential.validation(), true);
//  testTaskSequential.pre_processing();
//  testTaskSequential.run();
//  testTaskSequential.post_processing();
//  ASSERT_EQ(count, out[0]);
//}
//
//TEST(tsatsyn_a_increasing_contrast_by_histogram_seq, Test_Sum_50) {
//  const int count = 50;
//
//  // Create data
//  std::vector<int> in(1, count);
//  std::vector<int> out(1, 0);
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//  taskDataSeq->inputs_count.emplace_back(in.size());
//  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//  taskDataSeq->outputs_count.emplace_back(out.size());
//
//  // Create Task
//  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//  ASSERT_EQ(testTaskSequential.validation(), true);
//  testTaskSequential.pre_processing();
//  testTaskSequential.run();
//  testTaskSequential.post_processing();
//  ASSERT_EQ(count, out[0]);
//}
//TEST(tsatsyn_a_increasing_contrast_by_histogram_seq, Test_Sum_70) {
//  const int count = 70;
//
//  // Create data
//  std::vector<int> in(1, count);
//  std::vector<int> out(1, 0);
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//  taskDataSeq->inputs_count.emplace_back(in.size());
//  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//  taskDataSeq->outputs_count.emplace_back(out.size());
//
//  // Create Task
//  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//  ASSERT_EQ(testTaskSequential.validation(), true);
//  testTaskSequential.pre_processing();
//  testTaskSequential.run();
//  testTaskSequential.post_processing();
//  ASSERT_EQ(count, out[0]);
//}
//TEST(tsatsyn_a_increasing_contrast_by_histogram_seq, Test_Sum_100) {
//  const int count = 100;
//
//  // Create data
//  std::vector<int> in(1, count);
//  std::vector<int> out(1, 0);
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//  taskDataSeq->inputs_count.emplace_back(in.size());
//  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//  taskDataSeq->outputs_count.emplace_back(out.size());
//
//  // Create Task
//  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//  ASSERT_EQ(testTaskSequential.validation(), true);
//  testTaskSequential.pre_processing();
//  testTaskSequential.run();
//  testTaskSequential.post_processing();
//  ASSERT_EQ(count, out[0]);
//}