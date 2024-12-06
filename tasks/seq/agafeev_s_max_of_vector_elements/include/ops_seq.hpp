#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_max_of_vector_elements_seq {

template <typename T>
std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}

template <typename T>
T get_MaxValue(std::vector<T> matrix) {
  T max_result = std::numeric_limits<T>::min();
  for (unsigned int i = 0; i < matrix.size(); i++)
    if (max_result < matrix[i]) max_result = matrix[i];

  return max_result;
}

template <typename T>
class MaxMatrixSequental : public ppc::core::Task {
 public:
  explicit MaxMatrixSequental(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input_;
  T maxres_{};
};

}  // namespace agafeev_s_max_of_vector_elements_seq