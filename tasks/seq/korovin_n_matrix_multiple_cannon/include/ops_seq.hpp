#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace korovin_n_matrix_multiple_cannon_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    std::srand(std::time(nullptr));
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  int numRowsA_;
  int numColsA_RowsB_;
  int numColsB_;
};

}  // namespace korovin_n_matrix_multiple_cannon_seq