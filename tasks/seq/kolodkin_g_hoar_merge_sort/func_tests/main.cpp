// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/kolodkin_g_hoar_merge_sort/include/ops_seq.hpp"

namespace kolodkin_g_random_function {
std::vector<int> create_random_vector(unsigned n) {
  std::vector<int> vector;
  for (unsigned i = 0; i < n; i++) {
    vector.push_back(-100 + rand() % 100000);
  }
  return vector;
}
};  // namespace kolodkin_g_random_function

TEST(kolodkin_g_hoar_merge_sort_seq, Test_vector_with_one_elems) {
  // Create data
  std::vector<int> vector;
  std::vector<int> reference_out(1, 0);

  // Create TaskData
  vector = kolodkin_g_random_function::create_random_vector(1);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  auto vec_ptr = std::make_shared<std::vector<int>>(reference_out);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(vec_ptr.get()));

  kolodkin_g_hoar_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kolodkin_g_hoar_merge_sort_seq, Test_vector_with_two_elems) {
  // Create data
  std::vector<int> vector;
  std::vector<int> reference_out(2, 0);

  // Create TaskData
  vector = {50, 14};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  auto vec_ptr = std::make_shared<std::vector<int>>(reference_out);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(vec_ptr.get()));

  kolodkin_g_hoar_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  std::sort(vector.begin(), vector.end());
  for (unsigned i = 0; i < vector.size(); i++) {
    ASSERT_EQ(reference_out[i], vector[i]);
  }
}

TEST(kolodkin_g_hoar_merge_sort_seq, Test_vector_with_three_elems) {
  // Create data
  std::vector<int> vector;
  std::vector<int> reference_out(3, 0);

  // Create TaskData
  vector = {50, 14, 1000};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  auto vec_ptr = std::make_shared<std::vector<int>>(reference_out);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(vec_ptr.get()));

  kolodkin_g_hoar_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  std::sort(vector.begin(), vector.end());
  for (unsigned i = 0; i < vector.size(); i++) {
    ASSERT_EQ(reference_out[i], vector[i]);
  }
}

TEST(kolodkin_g_hoar_merge_sort_seq, Test_vector_with_negative_elems) {
  // Create data
  std::vector<int> vector;
  std::vector<int> reference_out(4, 0);

  // Create TaskData
  vector = {50, 14, -105, 0};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  auto vec_ptr = std::make_shared<std::vector<int>>(reference_out);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(vec_ptr.get()));

  kolodkin_g_hoar_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  std::sort(vector.begin(), vector.end());
  for (unsigned i = 0; i < vector.size(); i++) {
    ASSERT_EQ(reference_out[i], vector[i]);
  }
}

TEST(kolodkin_g_hoar_merge_sort_seq, Test_big_vector) {
  // Create data
  std::vector<int> vector;
  std::vector<int> reference_out(1000, 0);

  // Create TaskData
  vector = kolodkin_g_random_function::create_random_vector(1000);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  auto vec_ptr = std::make_shared<std::vector<int>>(reference_out);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(vec_ptr.get()));

  kolodkin_g_hoar_merge_sort_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  std::sort(vector.begin(), vector.end());
  for (unsigned i = 0; i < vector.size(); i++) {
    ASSERT_EQ(reference_out[i], vector[i]);
  }
}
