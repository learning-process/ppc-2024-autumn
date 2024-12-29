// Copyright 2023 Nesterov Alexander
#pragma once

#include <limits>
#include <stack>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace matyunina_a_batcher_qsort_seq {

template <typename T>
void quickSort(std::vector<T>& data);

template <typename T>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> data_;
};

}  // namespace matyunina_a_batcher_qsort_seq

template <typename T>
bool matyunina_a_batcher_qsort_seq::TestTaskSequential<T>::pre_processing() {
  internal_order_test();

  auto* input = reinterpret_cast<T*>(taskData->inputs[0]);
  data_.assign(input, input + taskData->inputs_count[0]);

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_seq::TestTaskSequential<T>::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == taskData->outputs_count[0]);
}

template <typename T>
bool matyunina_a_batcher_qsort_seq::TestTaskSequential<T>::run() {
  internal_order_test();

  if (data_.empty()) {
    return true;
  }

  quickSort(data_);

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_seq::TestTaskSequential<T>::post_processing() {
  internal_order_test();

  std::copy(data_.begin(), data_.end(), reinterpret_cast<T*>(taskData->outputs[0]));

  return true;
}

template <typename T>
void matyunina_a_batcher_qsort_seq::quickSort(std::vector<T>& data) {
  if (data.empty()) return;

  std::stack<std::pair<uint32_t, uint32_t>> stack;
  stack.emplace(0, data.size() - 1);

  uint32_t low;
  uint32_t high;
  T pivot;
  uint32_t i;
  std::pair<uint32_t, uint32_t> range;
  while (!stack.empty()) {
    range = stack.top();
    stack.pop();

    low = range.first;
    high = range.second;

    if (low < high) {
      pivot = data[high];
      i = low;

      for (uint32_t j = low; j < high; j++) {
        if (data[j] <= pivot) {
          if (i != j) {
            std::swap(data[i], data[j]);
          }
          i++;
        }
      }
      std::swap(data[i], data[high]);

      if (i > 0 && low < i - 1) {
        stack.emplace(low, i - 1);
      }
      if (i + 1 < high) {
        stack.emplace(i + 1, high);
      }
    }
  }
}
