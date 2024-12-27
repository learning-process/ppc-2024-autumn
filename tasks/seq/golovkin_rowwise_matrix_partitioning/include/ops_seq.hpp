// Golovkin Maksim Task#2

#pragma once

#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace golovkin_rowwise_matrix_partitioning {

class MatrixMultiplicationTask : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationTask(const std::shared_ptr<ppc::core::TaskData>& taskData);

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  static bool multiplier(std::vector<std::vector<double>>& matrixA, std::vector<std::vector<double>>& matrixB,
                         std::vector<std::vector<double>>& result);

  std::shared_ptr<ppc::core::TaskData> taskData_;

  std::vector<std::vector<double>> result_;
};

}  // namespace golovkin_rowwise_matrix_partitioning