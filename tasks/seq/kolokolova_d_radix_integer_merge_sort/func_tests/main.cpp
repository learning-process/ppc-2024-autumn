// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "seq/kolokolova_d_radix_integer_merge_sort/include/ops_seq.hpp"

namespace kolokolova_d_radix_integer_merge_sort_seq {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(-10000, 10000);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace kolokolova_d_radix_integer_merge_sort_seq

TEST(kolokolova_d_radix_integer_merge_sort_seq, Test_Sort1) {
  std::vector<int> unsorted_vector = {2000, 6, 100, 53, 234, 2};
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result = {2, 6, 53, 100, 234, 2000};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
  taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
  taskDataSeq->outputs_count.emplace_back(sorted_vector.size());

  kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(sorted_vector, result);
}

TEST(kolokolova_d_radix_integer_merge_sort_seq, Test_Sort2) {
  std::vector<int> unsorted_vector = {-1034, -3, -25, -200, -72, -1};
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result = {-1034, -200, -72, -25, -3, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
  taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
  taskDataSeq->outputs_count.emplace_back(sorted_vector.size());

  kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(sorted_vector, result);
}

TEST(kolokolova_d_radix_integer_merge_sort_seq, Test_Sort3) {
  std::vector<int> unsorted_vector = {239, -120, 567, 1, 23, -9, 1000};
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result = {-120, -9, 1, 23, 239, 567, 1000};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
  taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
  taskDataSeq->outputs_count.emplace_back(sorted_vector.size());

  kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(sorted_vector, result);
}

TEST(kolokolova_d_radix_integer_merge_sort_seq, Test_Sort4) {
  std::vector<int> unsorted_vector = {10000, -10000, 235, 0, -387, 235, 235, -387};
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result = {-10000, -387, -387, 0, 235, 235, 235, 10000};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
  taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
  taskDataSeq->outputs_count.emplace_back(sorted_vector.size());

  kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(sorted_vector, result);
}