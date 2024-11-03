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

  if (rows_ <= 0 || cols_ <= 0) {
    // std::cerr << "Error: Matrix dimensions must be positive." << std::endl;
    return false;
  }

  auto inputMatrixData = reinterpret_cast<T*>(taskData->inputs[1]);
  matrix.assign(inputMatrixData, inputMatrixData + rows_ * cols_);

  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::validation() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->outputs.empty()) {
    std::cerr << "Validation error: Missing input or output data." << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::run() {
  internal_order_test();

  res = get_max_matrix_element(matrix);

  auto outputData = reinterpret_cast<T*>(taskData->outputs[0]);
  outputData[0] = res;

  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::post_processing() {
  internal_order_test();
  // std::cout << "Maximum value in the matrix: " << res << std::endl;
  return true;
}

}  // namespace belov_a_max_value_of_matrix_elements_seq

#endif  // OPS_SEQ_HPP