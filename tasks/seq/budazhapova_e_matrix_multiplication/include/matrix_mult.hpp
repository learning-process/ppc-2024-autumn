#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_e_matrix_mult_seq {

class MatrixMultSequential : public ppc::core::Task {
 public:
  explicit MatrixMultSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows{};
  int columns{};

  std::vector<int> A;
  std::vector<int> b;
  std::vector<int> res;
};
}  // namespace budazhapova_e_matrix_mult_seq