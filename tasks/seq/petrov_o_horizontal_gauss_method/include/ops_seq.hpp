#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_horizontal_gauss_method_seq {

class GaussHorizontalSequential : public ppc::core::Task {
 public:
  explicit GaussHorizontalSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix;
  std::vector<double> b;
  std::vector<double> x;
};

}  // namespace petrov_o_horizontal_gauss_method_seq