#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_max_of_vector_elements_sequental {

template <typename Type>
std::vector<Type> create_RandomMatrix(Type row_size, Type column_size);

template <typename Type>
Type get_MaxValue(std::vector<Type> matrix);

template <typename Type>
class MaxMatrixSequential : public ppc::core::Task {
 public:
  explicit MaxMatrixSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Type> input_;
  Type maxres_{};
};

}  // namespace agafeev_s_max_of_vector_elements_sequental