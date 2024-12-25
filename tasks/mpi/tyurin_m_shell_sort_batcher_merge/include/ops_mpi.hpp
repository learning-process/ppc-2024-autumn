#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace tyurin_m_shell_sort_batcher_merge_mpi {

template <typename RandomIt>
void shellSort(RandomIt begin, RandomIt end) {
  auto n = std::distance(begin, end);
  if (n <= 1) return;

  std::vector<int> gaps;
  int k = 0;
  int gap;
  do {
    if (k % 2 == 0) {
      gap = 9 * std::pow(2, k) - 9 * std::pow(2, k / 2) + 1;
    } else {
      gap = 8 * std::pow(2, k) - 6 * std::pow(2, (k + 1) / 2) + 1;
    }
    if (gap < n) gaps.push_back(gap);
    k++;
  } while (gap < n);

  std::sort(gaps.rbegin(), gaps.rend());
  for (int gap2 : gaps) {
    for (auto i = begin + gap2; i != end; ++i) {
      auto temp = *i;
      auto j = i;
      while (j >= begin + gap2 && *(j - gap2) > temp) {
        *j = *(j - gap2);
        j -= gap2;
      }
      *j = temp;
    }
  }
}

template <typename RandomIt1, typename RandomIt2>
void swapSortedArrays(RandomIt1 begin1, RandomIt1 end1, RandomIt2 begin2, RandomIt2 end2) {
  auto size = std::distance(begin1, end1);
  std::vector<typename std::iterator_traits<RandomIt1>::value_type> temp(size * 2);
  auto it1 = begin1;
  auto it2 = begin2;
  auto tempIt = temp.begin();

  while (it1 != end1 && it2 != end2) {
    if (*it1 <= *it2) {
      *tempIt++ = *it1++;
    } else {
      *tempIt++ = *it2++;
    }
  }
  while (it1 != end1) *tempIt++ = *it1++;
  while (it2 != end2) *tempIt++ = *it2++;

  std::copy(temp.begin(), temp.begin() + size, begin1);
  std::copy(temp.begin() + size, temp.end(), begin2);
}

inline void merge_sorted_parts(std::vector<int>& arr, int parts) {
  int n = arr.size();
  int part_size = n / parts;

  for (int step = part_size; step < n; step *= 2) {
    for (int left = 0; left < n; left += 2 * step) {
      int mid = std::min(left + step - 1, n - 1);
      int right = std::min(left + 2 * step - 1, n - 1);
      int n1 = mid - left + 1;
      int n2 = right - mid;

      std::vector<int> leftArr(n1);
      std::vector<int> rightArr(n2);

      for (int i = 0; i < n1; ++i) leftArr[i] = arr[left + i];
      for (int i = 0; i < n2; ++i) rightArr[i] = arr[mid + 1 + i];

      int i = 0;
      int j = 0;
      int k = left;

      while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
          arr[k] = leftArr[i];
          ++i;
        } else {
          arr[k] = rightArr[j];
          ++j;
        }
        ++k;
      }

      while (i < n1) {
        arr[k] = leftArr[i];
        ++i;
        ++k;
      }

      while (j < n2) {
        arr[k] = rightArr[j];
        ++j;
        ++k;
      }
    }
  }
}

inline void shell_sort_batcher_merge(const boost::mpi::communicator& world, std::vector<int>& local_data) {
  shellSort(local_data.begin(), local_data.end());

  for (int step = world.size() / 2; step > 0; step /= 2) {
    if (world.rank() % (step * 2) < step) {
      std::vector<int> other(local_data.size());
      world.recv(world.rank() + step, 0, other.data(), other.size());
      swapSortedArrays(local_data.begin(), local_data.end(), other.begin(), other.end());
      world.send(world.rank() + step, 1, other.data(), other.size());
    } else {
      int target = world.rank() - step;
      world.send(target, 0, local_data.data(), local_data.size());
      world.recv(target, 1, local_data.data(), local_data.size());
    }
  }
}

class ShellSortBatcherMerge : public ppc::core::Task {
 public:
  explicit ShellSortBatcherMerge(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n;
  std::vector<int> input_vector;
  std::vector<int> result;
  boost::mpi::communicator world;
};

}  // namespace tyurin_m_shell_sort_batcher_merge_mpi