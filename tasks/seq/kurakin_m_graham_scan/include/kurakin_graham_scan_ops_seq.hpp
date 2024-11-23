#pragma once

#include <cstring>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_graham_scan_seq {

bool isLeftAngle(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_point{};
  std::vector<std::vector<double>> input_;
  int count_res_point{};
  std::vector<double> x_res;
  std::vector<double> y_res;
};

}  // namespace kurakin_m_graham_scan_seq