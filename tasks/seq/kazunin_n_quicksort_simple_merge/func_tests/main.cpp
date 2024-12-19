#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/kazunin_n_quicksort_simple_merge/include/ops_seq.hpp"

namespace kazunin_n_quicksort_simple_merge_seq {

void template_test(const std::vector<int>& input_data) {
  std::vector<int> data = input_data;
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataPar->inputs_count.emplace_back(data.size());

  result_data.resize(data.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataPar->outputs_count.emplace_back(result_data.size());

  auto taskParallel = std::make_shared<QuicksortSimpleMergeSeq>(taskDataPar);

  if (taskParallel->validation()) {
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

}  // namespace kazunin_n_quicksort_simple_merge_seq

TEST(kazunin_n_quicksort_simple_merge_seq, test_sorted_ascending) {
  kazunin_n_quicksort_simple_merge_seq::template_test({1, 2, 3, 4, 5, 6, 8, 9, 5, 4, 3, 2, 1});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_almost_sorted_random) {
  kazunin_n_quicksort_simple_merge_seq::template_test({9, 7, 5, 3, 1, 2, 4, 6, 8, 10});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_sorted_descending) {
  kazunin_n_quicksort_simple_merge_seq::template_test({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_all_equal_elements) {
  kazunin_n_quicksort_simple_merge_seq::template_test({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_negative_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_mixed_order) {
  kazunin_n_quicksort_simple_merge_seq::template_test({1, 3, 2, 4, 6, 5, 7, 9, 8, 10});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_single_element) {
  std::vector<int> vec = {42};
  kazunin_n_quicksort_simple_merge_seq::template_test(vec);
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_empty_vector) {
  kazunin_n_quicksort_simple_merge_seq::template_test({});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_mixed_large_and_small_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({100, 99, 98, 1, 2, 3, 4, 5});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_positive_and_negative_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({-5, -10, 5, 10, 0});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_large_random_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({123, 456, 789, 321, 654, 987});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_random_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({9, 1, 4, 7, 2, 8, 5, 3, 6});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_max_and_min_int) {
  kazunin_n_quicksort_simple_merge_seq::template_test(
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_prime_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({11, 13, 17, 19, 23, 29, 31});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_descending_with_negatives) {
  kazunin_n_quicksort_simple_merge_seq::template_test({50, 40, 30, 20, 10, 0, -10, -20, -30});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_pi_digits) {
  kazunin_n_quicksort_simple_merge_seq::template_test({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_consecutive_mixed_numbers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({12, 14, 16, 18, 20, 15, 17, 19, 21});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_descending_with_zero_and_negatives) {
  kazunin_n_quicksort_simple_merge_seq::template_test({7, 6, 5, 4, 3, 2, 1, 0, -1, -2});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_large_integers) {
  kazunin_n_quicksort_simple_merge_seq::template_test({1000, 2000, 1500, 2500, 1750});
}

TEST(kazunin_n_quicksort_simple_merge_seq, test_mixed_digits_and_zero) {
  kazunin_n_quicksort_simple_merge_seq::template_test({8, 4, 2, 6, 1, 9, 5, 3, 7, 0});
}
