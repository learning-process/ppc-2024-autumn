#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_broadcast {

std::vector<double> getRandomPoints(int count);

class DistanceMatrixTaskSequential : public ppc::core::Task {
 public:
  explicit DistanceMatrixTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> points_;
  std::vector<double> distance_matrix_;
};

}  // namespace borisov_s_broadcast