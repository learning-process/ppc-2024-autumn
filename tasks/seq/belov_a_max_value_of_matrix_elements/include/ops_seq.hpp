#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace belov_a_max_value_of_matrix_elements_seq {

template <typename T>
class MaxValueOfMatrixElementsSequential : public ppc::core::Task {
 public:
  explicit MaxValueOfMatrixElementsSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows_ = 0;
  int cols_ = 0;
  T res{};
  std::vector<T> matrix;

  static T get_max_matrix_element(const std::vector<T>& matrix);
};

template <typename T>
T MaxValueOfMatrixElementsSequential<T>::get_max_matrix_element(const std::vector<T>& matrix) {
  return *std::max_element(matrix.begin(), matrix.end());
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::pre_processing() {
  internal_order_test();

  auto* dimensions = reinterpret_cast<int*>(taskData->inputs[0]);
  rows_ = dimensions[0];
  cols_ = dimensions[1];

  auto inputMatrixData = reinterpret_cast<T*>(taskData->inputs[1]);
  matrix.assign(inputMatrixData, inputMatrixData + rows_ * cols_);

  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::validation() {
  internal_order_test();

  return !taskData->inputs.empty() && reinterpret_cast<int*>(taskData->inputs[0])[0] > 0 &&
         reinterpret_cast<int*>(taskData->inputs[0])[1] > 0;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::run() {
  internal_order_test();

  res = get_max_matrix_element(matrix);
  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::post_processing() {
  internal_order_test();

  reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
  return true;
}

}  // namespace belov_a_max_value_of_matrix_elements_seq

#endif  // OPS_SEQ_HPP