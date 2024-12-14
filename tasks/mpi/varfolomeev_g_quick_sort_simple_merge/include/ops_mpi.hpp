// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_quick_sort_simple_merge_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static std::vector<int> quickSort(const std::vector<int>& vec) {
    std::vector<int> sortedvec = vec;
    quickSortRecursive(sortedvec, 0, sortedvec.size() - 1);
    return sortedvec;
  }

 private:
  static void quickSortRecursive(std::vector<int>& vec, int left, int right) {
    if (left >= right) return;
    int p = vec[(left + right) / 2];
    int i = left;
    int j = right;
    while (i <= j) {
      while (vec[i] < p) i++;
      while (vec[j] > p) j--;
      if (i <= j) {
        int tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
        i++;
        j--;
      }
    }
    quickSortRecursive(vec, left, j);
    quickSortRecursive(vec, i, right);
  }

  std::vector<int> input_;
  std::vector<int> res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  std::vector<int> res;
  boost::mpi::communicator world;

  // Вспомогательная функция для сортировки
  static std::vector<int> quickSortRecursive(const std::vector<int>& vec) {
    if ((int)vec.size() <= 1) {
      return vec;
    }

    int p = vec[(int)vec.size() / 2];

    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> equal;
    for (int i = 0; i < (int)vec.size(); ++i) {
      if (vec[i] < p) {
        left.emplace_back(vec[i]);
      }
      if (vec[i] > p) {
        right.emplace_back(vec[i]);
      }
      if (vec[i] == p) {
        equal.emplace_back(vec[i]);
      }
    }
    std::vector<int> resLeft = quickSortRecursive(left);
    std::vector<int> resRight = quickSortRecursive(right);

    std::vector<int> mergeEqualRight = merge(equal, resRight);
    return merge(resLeft, mergeEqualRight);
  }

  // Вспомогательная функция для слияния двух отсортированных массивов
  static std::vector<int> merge(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    std::vector<int> mergedvec;
    mergedvec.reserve(vec1.size() + vec2.size());
    int i = 0;
    int j = 0;
    while (i < (int)vec1.size() && j < (int)vec2.size()) {
      if (vec1[i] < vec2[j]) {
        mergedvec.push_back(vec1[i++]);
      } else {
        mergedvec.push_back(vec2[j++]);
      }
    }

    mergedvec.insert(mergedvec.end(), vec1.begin() + i, vec1.end());
    mergedvec.insert(mergedvec.end(), vec2.begin() + j, vec2.end());

    return mergedvec;
  }
};

}  // namespace varfolomeev_g_quick_sort_simple_merge_mpi
