#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace sarafanov_m_quick_sort_batcher_merge_mpi {

template <typename RandomIt, typename Compare = std::less<typename RandomIt::value_type>>
void quickSort(RandomIt begin, RandomIt end, Compare comp = Compare()) {
  const auto size = std::distance(begin, end);
  if (size <= 1) return;
  const int threshold = 10;
  if (size < threshold) {
    for (auto i = begin + 1; i != end; ++i) {
      auto key = *i;
      auto j = i;
      while (j > begin && comp(key, *(j - 1))) {
        *j = *(j - 1);
        --j;
      }
      *j = key;
    }
    return;
  }
  auto pivot = *(begin + size / 2);
  auto left = begin;
  auto right = end - 1;
  while (true) {
    while (comp(*left, pivot)) ++left;
    while (comp(pivot, *right)) --right;
    if (left >= right) break;
    std::iter_swap(left, right);
    ++left;
    --right;
  }
  quickSort(begin, right + 1, comp);
  quickSort(right + 1, end, comp);
}

void make_bitonic_sequence(std::vector<int>& arr);
void merge_batcher(boost::mpi::communicator& world, std::vector<int>& arr, int arr_size);

class QuicksortBatcherMerge : public ppc::core::Task {
 public:
  explicit QuicksortBatcherMerge(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int vector_size;
  std::vector<int> arr;
  boost::mpi::communicator world;
};

}  // namespace sarafanov_m_quick_sort_batcher_merge_mpi
