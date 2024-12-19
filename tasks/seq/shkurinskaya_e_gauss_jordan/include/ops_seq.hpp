#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_gauss_jordan_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n = 0;
  std::vector<double> matrix;
  std::vector<double> solution;
};

}  // namespace shkurinskaya_e_gauss_jordan_seq
