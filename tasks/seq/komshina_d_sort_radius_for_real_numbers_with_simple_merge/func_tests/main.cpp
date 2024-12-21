#include <gtest/gtest.h>

#include <vector>

#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, SortRegularNumbers) {
  std::vector<double> input = {1.2, -2.5, 3.6, -1.1, 2.8, -3.3};
  std::vector<double> expected = {-3.3, -2.5, -1.1, 1.2, 2.8, 3.6};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  std::vector<double> output(input.size(), 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(expected, output);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, SortEmptyInput) {
  std::vector<double> input = {};
  std::vector<double> expected = {};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  std::vector<double> output(input.size(), 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(expected, output);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, SortSingleElement) {
  std::vector<double> input = {4.5};
  std::vector<double> expected = {4.5};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  std::vector<double> output(input.size(), 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(expected, output);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, SortAlreadySortedInput) {
  std::vector<double> input = {-3.3, -2.5, -1.1, 1.2, 2.8, 3.6};
  std::vector<double> expected = {-3.3, -2.5, -1.1, 1.2, 2.8, 3.6};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  std::vector<double> output(input.size(), 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(expected, output);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, SortLargeNumbers) {
  std::vector<double> input = {1.2e10, 5.5e8, -2.3e12, -1.1e9, 3.0e7, -8.8e5};
  std::vector<double> expected = {-2.3e12, -1.1e9, -8.8e5, 3.0e7, 5.5e8, 1.2e10};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  std::vector<double> output(input.size(), 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(expected, output);
}