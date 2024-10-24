#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_sum_element_matrix_seq {

template <typename T>
std::vector<T> create_random_matrix(int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    return {};
  }

  std::vector<T> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-100, 100);

  std::generate(matrix.begin(), matrix.end(), [&]() { return static_cast<T>(dis(gen)); });
  return matrix;
}

template <typename T>
T sum_matrix_elements(const std::vector<T>& matrix, int rows, int cols) {
  return std::accumulate(matrix.begin(), matrix.end(), T(0));
}

template <typename T>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input_data_;
  T result_{};
};

template <typename T>
bool TestTaskSequential<T>::pre_processing() {
  internal_order_test();
  result_ = 0;
  T* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
  input_data_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  return true;
}

template <typename T>
bool TestTaskSequential<T>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

template <typename T>
bool TestTaskSequential<T>::run() {
  internal_order_test();
  result_ = std::accumulate(input_data_.begin(), input_data_.end(), T(0));
  return true;
}

template <typename T>
bool TestTaskSequential<T>::post_processing() {
  internal_order_test();
  if (!taskData->outputs.empty() && taskData->outputs[0] != nullptr) {
    reinterpret_cast<T*>(taskData->outputs[0])[0] = result_;
    return true;
  }
  return false;
}

template class sotskov_a_sum_element_matrix_seq::TestTaskSequential<int>;
template class sotskov_a_sum_element_matrix_seq::TestTaskSequential<double>;

}  // namespace sotskov_a_sum_element_matrix_seq
