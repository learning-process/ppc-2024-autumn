#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/yasakova_t_quick_sort_with_simple_merge/include/ops_seq.hpp"

namespace yasakova_t_quick_sort_with_simple_merge_seq {

void executeQuickSortTest(const std::vector<int>& inputData) {
  std::vector<int> unsortedData = inputData;
  std::vector<int> sortedData;

  std::shared_ptr<ppc::core::TaskData> taskDataPtr = std::make_shared<ppc::core::TaskData>();

  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t*>(unsortedData.data()));
  taskDataPtr->inputs_count.emplace_back(unsortedData.size());

  sortedData.resize(unsortedData.size());
  taskDataPtr->outputs.emplace_back(reinterpret_cast<uint8_t*>(sortedData.data()));
  taskDataPtr->outputs_count.emplace_back(sortedData.size());

  auto quickSortTask =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_seq::QuickSortWithMergeSeq>(taskDataPtr);

  if (quickSortTask->validation()) {
    quickSortTask->pre_processing();
    quickSortTask->run();
    quickSortTask->post_processing();

    std::sort(unsortedData.begin(), unsortedData.end());
    EXPECT_EQ(unsortedData, sortedData);
  }
}
}  // namespace yasakova_t_quick_sort_with_simple_merge_seq

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesSortedInput) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({1, 2, 3, 4, 5, 6, 8, 9, 5, 4, 3, 2, 1});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, Test_almost_sorted_random) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({9, 7, 5, 3, 1, 2, 4, 6, 8, 10});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesReverseSortedInput) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesAllEqualElements) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesAlmostSortedInput) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({1, 3, 2, 4, 6, 5, 7, 9, 8, 10});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesSingleElementInput) {
  std::vector<int> vec = {42};
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest(vec);
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesEmptyInput) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesMixedLargeAndSmallNumbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({100, 99, 98, 1, 2, 3, 4, 5});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesPositiveAndNegativeNumbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({-5, -10, 5, 10, 0});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesLargeRandomNumbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({123, 456, 789, 321, 654, 987});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesRandomNumbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({9, 1, 4, 7, 2, 8, 5, 3, 6});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesMaxAndMinIntValues) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest(
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesPrimeNumbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({11, 13, 17, 19, 23, 29, 31});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesDescendingOrderWithNegatives) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({50, 40, 30, 20, 10, 0, -10, -20, -30});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesPiDigits) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesConsecutiveMixedNumbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({12, 14, 16, 18, 20, 15, 17, 19, 21});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesDescendingOrderWithZeroAndNegatives) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({7, 6, 5, 4, 3, 2, 1, 0, -1, -2});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesLargeIntegers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({1000, 2000, 1500, 2500, 1750});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, HandlesMixedDigitsAndZero) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({8, 4, 2, 6, 1, 9, 5, 3, 7, 0});
}

TEST(yasakova_t_quick_sort_with_simple_merge_seq, Test_negative_numbers) {
  yasakova_t_quick_sort_with_simple_merge_seq::executeQuickSortTest({0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
}