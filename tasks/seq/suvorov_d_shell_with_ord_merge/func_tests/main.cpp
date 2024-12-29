// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/suvorov_d_shell_with_ord_merge/include/ops_seq.hpp"

namespace suvorov_d_shell_with_ord_merge_seq {
void check_correctness_cases(const std::vector<int> &correct_test_data) {
  std::vector<int> data_to_sort = correct_test_data;
  size_t count_of_elems = data_to_sort.size();
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(sorted_result.size());

  suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
  ASSERT_EQ(ShellSortSeq.validation(), true);
  ShellSortSeq.pre_processing();
  ShellSortSeq.run();
  ShellSortSeq.post_processing();
  EXPECT_TRUE(std::is_sorted(sorted_result.begin(), sorted_result.end()));
}
}  // namespace suvorov_d_shell_with_ord_merge_seq

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_with_unique_elements) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({5, 2, 7, 4, 1, 3, 9});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_with_repeating_elements) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({5, 2, 4, 8, 5, 1, 2, 2, 5, 9, 8});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_with_a_single_element) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({31});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_with_negative_elements) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({5, -2, 4, 7, -12, 10, 3, -5});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_an_already_sorted_vector) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({2, 4, 7, 9, 12, 31, 42, 63, 85});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_sorted_in_reverse_order) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({93, 78, 61, 45, 33, 20, 13, 9, 5, 2});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_with_identical_elements) {
  suvorov_d_shell_with_ord_merge_seq::check_correctness_cases({25, 25, 25, 25, 25, 25, 25, 25, 25});
}

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_an_empty_vector) {
  std::vector<int> data_to_sort = {};
  size_t count_of_elems = data_to_sort.size();
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(sorted_result.size());

  suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
  EXPECT_FALSE(ShellSortSeq.validation());
}

TEST(suvorov_d_shell_with_ord_merge_seq, Checking_an_attempt_to_transfer_a_zero_input_size_inside_the_task) {
  std::vector<int> data_to_sort = {5, -2, 4, 7, -12, 10, 3, -5};
  size_t count_of_elems = data_to_sort.size();
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(0);
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(sorted_result.size());

  suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
  EXPECT_FALSE(ShellSortSeq.validation());
}

TEST(suvorov_d_shell_with_ord_merge_seq, Checking_an_attempt_to_transfer_a_zero_output_size_inside_the_task) {
  std::vector<int> data_to_sort = {5, -2, 4, 7, -12, 10, 3, -5};
  size_t count_of_elems = data_to_sort.size();
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(0);

  suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
  EXPECT_FALSE(ShellSortSeq.validation());
}

TEST(suvorov_d_shell_with_ord_merge_seq,
     Checking_an_attempt_to_transfer_unequal_input_and_output_sizes_inside_the_task) {
  std::vector<int> data_to_sort = {5, -2, 4, 7, -12, 10, 3, -5};
  size_t count_of_elems = data_to_sort.size();
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(data_to_sort.size() + 1);

  suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
  EXPECT_FALSE(ShellSortSeq.validation());
}
