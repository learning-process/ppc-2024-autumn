#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_basic_sorting) {
  std::vector<double> inputData = {1.1, -6.9, 0.42, 0.0, 2.14, -2.13};
  std::vector<double> out(inputData.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int input_size = inputData.size();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_empty_array) {
  std::vector<double> inputData = {};
  std::vector<double> out(inputData.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int input_size = inputData.size();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_TRUE(out.empty());
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_same_elements) {
  std::vector<double> inputData = {5.0, 5.0, 5.0, 5.0};
  std::vector<double> out(inputData.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int input_size = inputData.size();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_reverse_sorted_array) {
  std::vector<double> inputData = {9.0, 7.5, 5.3, 2.2, 1.1};
  std::vector<double> out(inputData.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int input_size = inputData.size();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_negative_numbers) {
  std::vector<double> inputData = {-3.5, -2.1, -5.7, -1.0};
  std::vector<double> out(inputData.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int input_size = inputData.size();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}
