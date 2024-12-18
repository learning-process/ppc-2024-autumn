// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>

#include "seq/ermilova_d_Shell_sort_simple_merge/include/ops_seq.hpp"

static std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, Can_create_vector) {
  const int size_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, Cant_create_incorrect_vector) {
  const int size_test = -10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, Test_vec_10) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 10;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> res = input;
  std::sort(res.begin(), res.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  // Create Task
  ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(output, res);
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, Test_vec_100) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 100;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  // Create Task
  ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(output, sort_ref);
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, Test_vec_1000) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 1000;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  // Create Task
  ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(output, sort_ref);
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, Test_vec_10000) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 10000;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  // Create Task
  ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(output, sort_ref);
}