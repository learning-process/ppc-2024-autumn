// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "seq/varfolomeev_g_quick_sort_simple_merge/include/ops_seq.hpp"

namespace varfolomeev_g_quick_sort_simple_merge_seq {
static std::vector<int> getRandomVector_seq(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

static std::vector<int> getAntisorted_seq(int sz, int a) {  // (a, a + sz]
  if (sz <= 0) {
    return {};
  }
  std::vector<int> vec(sz);
  for (int i = a + sz, j = 0; i > a && j < sz; i--, j++) {
    vec[j] = i;
  }
  return vec;
}
}  // namespace varfolomeev_g_quick_sort_simple_merge_seq

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_getRandomVector_func) {
  int size = 100;
  int lower_bound = -50;
  int upper_bound = 50;
  std::vector<int> vec = varfolomeev_g_quick_sort_simple_merge_seq::getRandomVector_seq(size, lower_bound, upper_bound);
  EXPECT_EQ(static_cast<int>(vec.size()), size);
  for (int value : vec) {
    EXPECT_GE(value, lower_bound);
    EXPECT_LE(value, upper_bound);
  }
  std::vector<int> vec2 =
      varfolomeev_g_quick_sort_simple_merge_seq::getRandomVector_seq(size, lower_bound, upper_bound);
  EXPECT_NE(vec, vec2);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_getAntisortedVecor_func) {
  int size = 100;
  int start = -50;
  std::vector<int> vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, start);
  EXPECT_EQ(static_cast<int>(vec.size()), size);
  for (int i = 0; i < static_cast<int>(vec.size()) - 1; i++) {
    EXPECT_LE(vec[i + 1], vec[i]);
  }
  for (int value : vec) {
    EXPECT_LE(value, start + size + 1);
    EXPECT_GE(value, start);
  }
  EXPECT_TRUE(varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(0, 10).empty());
  EXPECT_TRUE(varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(-5, 10).empty());
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_manual_10) {
  // Create data
  std::vector<int> global_vec = {100, 23, 332, 67, -67, -45, 34, 0};
  std::vector<int> global_res(global_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  // Create Task
  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Zero) {
  std::vector<int> global_vec;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_EQ(testTaskSequential.validation(), false);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_wrong_output_size) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  global_res.resize(global_vec.size() + 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), false);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Single) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  global_vec = {33};
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Already_Sorted) {
  std::vector<int> global_vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Manual_Random_2) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  global_vec = {33, 23};
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Manual_Random_4) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  global_vec = {33, 23, 5, 44};
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_Positive) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 150;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_Negative) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 150;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, -200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_Positive_Negative) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 150;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, -75);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_RandomVector_Positive) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 150;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getRandomVector_seq(size, 0, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_RandomVector_Negative) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 150;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getRandomVector_seq(size, -200, 0);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_RandomVector_Positive_Negative) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 150;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getRandomVector_seq(size, -100, 100);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_64) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 64;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_128) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 128;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_512) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 512;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_1024) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 1024;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_4096) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 4096;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_8192) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 8192;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}

TEST(varfolomeev_g_quick_sort_simple_merge_seq, Test_Antisorted_65536) {
  std::vector<int> global_vec;
  std::vector<int> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int size = 65536;
  global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(size, 200);
  global_res.resize(global_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  EXPECT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  EXPECT_TRUE(isSorted);
}