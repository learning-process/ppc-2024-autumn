
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace Odintsov_M_VerticalRibbon_seq {

class VerticalRibbonSequential : public ppc::core::Task {
 public:
  explicit VerticalRibbonSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std ::vector<double> matrixA;
  std::vector<double> vectorB;
  std::vector<double> vectorC;
  // [0] - size, [1] - row, [2] - col
  int colA, rowA;
  int szA;
  int szB;
};

}  // namespace Odintsov_M_VerticalRibbon_seq