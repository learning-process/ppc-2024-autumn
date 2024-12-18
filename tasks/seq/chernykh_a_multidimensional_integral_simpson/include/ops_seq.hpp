#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_simpson_seq {

double integrate_1d(const std::function<double(double)> &func,  //
                    const std::pair<double, double> &bound,     //
                    int num_steps);

double integrate_nd(const std::function<double(const std::vector<double> &)> &func,  //
                    std::vector<double> &func_args,                                  //
                    const std::vector<std::pair<double, double>> &bounds,            //
                    const std::pair<int, int> &step_range,                           //
                    double tolerance,                                                //
                    int dim);

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::function<double(const std::vector<double> &)> func;
  std::vector<double> func_args;
  std::vector<std::pair<double, double>> bounds;
  std::pair<int, int> step_range;
  double tolerance{};
  double result{};
};

}  // namespace chernykh_a_multidimensional_integral_simpson_seq
