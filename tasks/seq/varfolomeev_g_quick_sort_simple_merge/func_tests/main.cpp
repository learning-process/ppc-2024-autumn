// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "seq/varfolomeev_g_quick_sort_simple_merge/include/ops_seq.hpp"

static std::vector<int> getRandomVector_(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

static std::vector<int> getAntisorted(int sz, int a) {  // [a + sz, a)
  std::vector<int> vec(sz);
  for (int i = a + sz, j = 0; i > a && j < sz; i--, j++) {
    vec[j] = i;
  }
  return vec;
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_manual_10) {
  // Create data
  std::vector<int> in = {100, 23, 332, 67, -67, -45, 34, 0};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_single) {
  // Create data
  std::vector<int> in = {333};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_zeros) {
  // Create data
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_150) {
  // Create data
  std::vector<int> in = getRandomVector_(150, -100, 100);
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_150_antiSorted_Positive) {
  // Create data
  std::vector<int> in = getAntisorted(150, 100);
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_150_antiSorted_Negative) {
  // Create data
  std::vector<int> in = getAntisorted(150, -200);
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_qSort_150_antiSorted_All) {
  // Create data
  std::vector<int> in = getAntisorted(150, -75);
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = true;
  for (int i = 0; i < out.size() - 1; i++) {
    if (out[i + 1] < out[i]) {
      isSorted = false;
    }
  }
  ASSERT_TRUE(isSorted);
}