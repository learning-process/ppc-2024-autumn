#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_sorting_5_elements) {
  std::vector<double> in = {5.1, -1.0, 2.2, 4.5, -3.3};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]) << "Array is not sorted at index " << i;
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_empty_array) {
  std::vector<double> in = {};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_single_element) {
  std::vector<double> in = {42.0};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_EQ(out[0], 42.0);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_identical_elements) {
  std::vector<double> in = {5.5, 5.5, 5.5, 5.5, 5.5};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], 5.5);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_extreme_values) {
  std::vector<double> in = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.0};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  ASSERT_LE(out[0], out[1]);
  ASSERT_LE(out[1], out[2]);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_near_zero_values) {
  std::vector<double> in = {0.0000001, -0.0000001, 0.0000005, -0.0000003, 0.0000002};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]) << "Array is not sorted at index " << i;
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_high_precision_values) {
  std::vector<double> in = {3.14159265358979323846264338327950288419716939937510,
                            2.71828182845904523536028747135266249775724709369995,
                            1.61803398874989484820458683436563811790028449733817};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]) << "Array is not sorted at index " << i;
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_large_array) {
  std::vector<double> in(1000000, 1.0);
  in[500000] = -1000.0;
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]) << "Array is not sorted at index " << i;
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_sorted_array) {
  std::vector<double> in = {-3.3, -2.2, 0.0, 1.1, 5.5};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], in[i]);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_all_negative_values) {
  std::vector<double> in = {-1.0, -2.0, -3.0, -4.0, -5.0};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]) << "Array is not sorted at index " << i;
  }
}