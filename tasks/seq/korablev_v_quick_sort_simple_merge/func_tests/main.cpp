#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "seq/korablev_v_quick_sort_simple_merge/include/ops_seq.hpp"

TEST(korablev_v_quick_sort_simple_merge, test_sort_array) {
  const size_t array_size = 6;
  std::vector<size_t> in_size(1, array_size);
  std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
  std::vector<int> expected_output = {2, 3, 5, 6, 7, 8};
  std::vector<int> out(array_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);

  ASSERT_TRUE(quickSortTask.validation());

  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

  for (size_t i = 0; i < array_size; ++i) {
    ASSERT_EQ(out[i], expected_output[i]);
  }
}

TEST(korablev_v_quick_sort_simple_merge, test_single_element) {
  const size_t array_size = 1;
  std::vector<size_t> in_size(1, array_size);
  std::vector<int> input_data = {42};
  std::vector<int> expected_output = {42};
  std::vector<int> out(array_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);

  ASSERT_TRUE(quickSortTask.validation());

  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

  for (size_t i = 0; i < array_size; ++i) {
    ASSERT_EQ(out[i], expected_output[i]);
  }
}

TEST(korablev_v_quick_sort_simple_merge, test_all_equal_elements) {
  const size_t array_size = 4;
  std::vector<size_t> in_size(1, array_size);
  std::vector<int> input_data = {7, 7, 7, 7};
  std::vector<int> expected_output = {7, 7, 7, 7};
  std::vector<int> out(array_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);

  ASSERT_TRUE(quickSortTask.validation());

  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

  for (size_t i = 0; i < array_size; ++i) {
    ASSERT_EQ(out[i], expected_output[i]);
  }
}

TEST(korablev_v_quick_sort_simple_merge, test_negative_and_positive) {
  const size_t array_size = 5;
  std::vector<size_t> in_size(1, array_size);
  std::vector<int> input_data = {-3, 2, -1, 0, 1};
  std::vector<int> expected_output = {-3, -1, 0, 1, 2};
  std::vector<int> out(array_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);

  ASSERT_TRUE(quickSortTask.validation());

  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

  for (size_t i = 0; i < array_size; ++i) {
    ASSERT_EQ(out[i], expected_output[i]);
  }
}

TEST(korablev_v_quick_sort_simple_merge, invalid_input_count) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<size_t> in_size(1, 6);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());

  std::vector<int> out(6, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_FALSE(quickSortTask.validation());
}

TEST(korablev_v_quick_sort_simple_merge, invalid_output_count) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<size_t> in_size(1, 6);
  std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_FALSE(quickSortTask.validation());
}

TEST(korablev_v_quick_sort_simple_merge, invalid_negative_size) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  auto invalid_size = static_cast<size_t>(-1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&invalid_size));
  taskDataSeq->inputs_count.emplace_back(1);

  std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());

  std::vector<int> out(6, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_FALSE(quickSortTask.validation());
}

TEST(korablev_v_quick_sort_simple_merge, invalid_size_data_mismatch) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t array_size = 6;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
  taskDataSeq->inputs_count.emplace_back(1);

  std::vector<int> input_data = {5, 3, 8};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());

  std::vector<int> out(6, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_FALSE(quickSortTask.validation());
}

TEST(korablev_v_quick_sort_simple_merge, invalid_output_size) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t array_size = 6;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
  taskDataSeq->inputs_count.emplace_back(1);

  std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());

  std::vector<int> out(3, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_FALSE(quickSortTask.validation());
}