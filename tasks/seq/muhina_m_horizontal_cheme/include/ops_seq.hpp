// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>
#include <memory>

#include "core/task/include/task.hpp"

namespace muhina_m_horizontal_cheme_seq {
std::vector<int> matrixVectorMultiplication(const std::vector<int>& matrix, const std::vector<int>& vec, int rows,
                                            int cols);

class HorizontalSchemeSequential : public ppc::core::Task {
 public:
  explicit HorizontalSchemeSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;  
  std::vector<int> vec_;     
  std::vector<int> result_;             
};
}  // namespace muhina_m_horizontal_cheme_seq