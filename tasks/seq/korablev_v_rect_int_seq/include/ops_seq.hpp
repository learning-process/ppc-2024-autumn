#pragma once
#include <functional>
#include <memory>

#include "core/task/include/task.hpp"

namespace korablev_v_rect_int_seq {

class RectangularIntegrationSequential : public ppc::core::Task {
 public:
  explicit RectangularIntegrationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  static double integrate(const std::function<double(double)>& f, double a, double b, int n);

  double a_{};
  double b_{};
  int n_{};
  double result_{};
  std::function<double(double)> func_;
};

}  // namespace korablev_v_rect_int_seq