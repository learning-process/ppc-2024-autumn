#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "seq/petrov_o_radix_sort_with_simple_merge/include/ops_seq.hpp"

TEST(petrov_o_radix_sort_with_simple_merge_seq, BasicSortTest) {
  std::vector<int> in{8, 3};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_seq, NegativeNumbersTest) {
  std::vector<int> in{-100, -5, -3, 2, 7, 12};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_seq, ReverseSortedTest) {
  std::vector<int> in{10, 8, 6, 4, 2, 0};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_seq, DuplicateElementsTest) {
  std::vector<int> in{5, 5, 5, 5, 5, 5};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}