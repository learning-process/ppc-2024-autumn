#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_max_of_vector_elements_sequental {

template <typename T>
std::vector<T> create_RandomMatrix(int row_size, int column_size);

template <typename T>
T get_MaxValue(std::vector<T> matrix);

template <typename T>
class MaxMatrixSequential : public ppc::core::Task {
 public:
  explicit MaxMatrixSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input_;
  T maxres_{};
};

}  // namespace agafeev_s_max_of_vector_elements_sequental