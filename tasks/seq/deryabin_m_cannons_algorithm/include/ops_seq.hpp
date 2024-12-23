#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_cannons_algorithm_seq {

class CannonsAlgorithmTaskSequential : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_matrix_A;
  std::vector<double> input_matrix_B;
  std::vector<double> output_matrix_C;
};

}  // namespace deryabin_m_cannons_algorithm_seq
