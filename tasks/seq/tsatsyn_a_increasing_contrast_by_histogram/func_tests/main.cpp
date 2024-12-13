// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/tsatsyn_a_increasing_contrast_by_histogram/include/ops_seq.hpp"

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_1) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {64, 67, 10, 152, 152, 106, 117, 10, 155, 78};
  std::vector<int> reference = {94, 100, 0, 249, 249, 168, 188, 0, 255, 119};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}
TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_2) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {224, 81, 34, 142, 14, 128, 125, 144, 175, 83};
  std::vector<int> reference = {255, 81, 24, 155, 0, 138, 134, 157, 195, 83};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}
TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_3) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {229, 97, 175, 70, 143, 90, 73, 81, 11, 53};
  std::vector<int> reference = {255, 100, 191, 69, 154, 92, 72, 81, 0, 49};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_4) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {75, 38, 133, 158, 66, 95, 137, 153, 178, 94};
  std::vector<int> reference = {67, 0, 173, 218, 51, 103, 180, 209, 255, 102};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_5) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {145, 85, 38, 40, 241, 161, 64, 94, 240, 57};
  std::vector<int> reference = {134, 59, 0, 2, 255, 154, 32, 70, 253, 23};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_6) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {251, 116, 160, 2, 152, 122, 165, 213, 149, 161};
  std::vector<int> reference = {255, 116, 161, 0, 153, 122, 166, 216, 150, 162};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_7) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {188, 71, 210, 3, 16, 213, 43, 219, 92, 22};
  std::vector<int> reference = {218, 80, 244, 0, 15, 247, 47, 255, 105, 22};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_8) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {59, 135, 129, 4, 28, 254, 74, 183, 15, 71};
  std::vector<int> reference = {56, 133, 127, 0, 24, 255, 71, 182, 11, 68};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_9) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {164, 103, 144, 17, 4, 185, 125, 234, 150, 131};
  std::vector<int> reference = {177, 109, 155, 14, 0, 200, 134, 255, 161, 140};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}

TEST(tsatsyn_a_increasing_contrast_by_histogram_mpi, Test_Sum_10) {
  // Create data
  std::vector<int> sizes = {10, 1};
  const int count_size_vector = sizes[0] * sizes[1];
  std::vector<int> in = {65, 78, 113, 107, 224, 0, 197, 103, 135, 96};
  std::vector<int> reference = {73, 88, 128, 121, 255, 0, 224, 117, 153, 109};
  std::vector<int> out(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, out);
}