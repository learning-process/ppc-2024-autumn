#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace kholin_k_multidimensional_integrals_rectangle_seq {
using Function = std::function<double(const std::vector<double>&)>;

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> f_values;
  Function f;
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
  double epsilon;
  int start_n;
  double result;

  size_t dim;
  size_t sz_values;
  size_t sz_lower_limits;
  size_t sz_upper_limits;

  double integrate(const Function& f_, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                   const std::vector<double>& h, std::vector<double>& f_values_, size_t curr_index_dim, size_t dim_,
                   size_t n);
  double integrate_with_rectangle_method(const Function& f_, std::vector<double>& f_values_,
                                         const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                         size_t dim_, size_t n);
  double run_multistep_scheme_method_rectangle(const Function& f_, std::vector<double>& f_values_,
                                               const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                               size_t dim_, double epsilon_, int n);
};

}  // namespace kholin_k_multidimensional_integrals_rectangle_seq