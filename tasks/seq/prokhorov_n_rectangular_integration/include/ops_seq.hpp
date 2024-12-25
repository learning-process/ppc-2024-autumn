#pragma once
#include <functional>
#include <memory>

#include "core/task/include/task.hpp"

namespace prokhorov_n_rectangular_integration {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  static double integrate(const std::function<double(double)>& f, double lower_bound, double upper_bound, int n);

  double lower_bound_;
  double upper_bound_;
  int n_;
  double result_;
  std::function<double(double)> f_;
};

}  // namespace prokhorov_n_rectangular_integration