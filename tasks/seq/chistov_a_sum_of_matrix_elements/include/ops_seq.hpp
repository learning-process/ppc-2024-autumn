#pragma once
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_a_sum_of_matrix_elements_seq {

template <typename T>
std::vector<T> get_random_matrix_seq(const int n, const int m) {
  if (n <= 0 || m <= 0) {
    return std::vector<T>();
  }

  std::vector<T> matrix(n * m);
  for (int i = 0; i < n * m; ++i) {
    matrix[i] = static_cast<T>((std::rand() % 201) - 100);
  }
  return matrix;
}

template <typename T>
T classic_way_seq(const std::vector<T> matrix, const int n, const int m) {
  T result = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      result += matrix[i * m + j];
    }
  }
  return result;
}

template <typename T>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input_;
  T res{};
};

}  // namespace chistov_a_sum_of_matrix_elements_seq
