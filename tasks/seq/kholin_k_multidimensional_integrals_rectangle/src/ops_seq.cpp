#include "seq/kholin_k_multidimensional_integrals_rectangle/include/ops_seq.hpp"

#include <thread>

double kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::integrate(
    const Function& f_, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double>& f_values_, size_t curr_index_dim, size_t dim_, size_t n) {
  if (curr_index_dim == dim_) {
    return f_(f_values_);
  }

  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    f_values_[curr_index_dim] = l_limits[curr_index_dim] + (i + 0.5) * h[curr_index_dim];
    sum += integrate(f_, l_limits, u_limits, h, f_values_, curr_index_dim + 1, dim_, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::integrate_with_rectangle_method(
    const Function& f_, std::vector<double>& f_values_, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim_, size_t n) {
  std::vector<double> h(dim_);
  for (size_t i = 0; i < dim_; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }

  return integrate(std::move(f_), l_limits, u_limits, h, f_values_, 0, dim_, n);
}

double kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::run_multistep_scheme_method_rectangle(
    const Function& f_, std::vector<double>& f_values_, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim_, double epsilon_, int n) {
  double I_n = integrate_with_rectangle_method(f_, f_values_, l_limits, u_limits, dim_, n);
  double I_2n;
  double delta = 0;
  do {
    n *= 2;
    I_2n = integrate_with_rectangle_method(f_, f_values_, l_limits, u_limits, dim_, n);
    delta = std::abs(I_2n - I_n);
    I_n = I_2n;

  } while ((1.0 / 3) * delta >= epsilon_);

  return I_2n;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  sz_values = taskData->inputs_count[0];
  sz_lower_limits = taskData->inputs_count[1];
  sz_upper_limits = taskData->inputs_count[2];

  auto* ptr_dim = reinterpret_cast<size_t*>(taskData->inputs[0]);
  dim = *ptr_dim;

  auto* ptr_f_values = reinterpret_cast<double*>(taskData->inputs[1]);
  f_values.assign(ptr_f_values, ptr_f_values + sz_values);

  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(taskData->inputs[2]);
  f = *ptr_f;

  auto* ptr_lower_limits = reinterpret_cast<double*>(taskData->inputs[3]);
  lower_limits.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits);

  auto* ptr_upper_limits = reinterpret_cast<double*>(taskData->inputs[4]);
  upper_limits.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits);

  auto* ptr_epsilon = reinterpret_cast<double*>(taskData->inputs[5]);
  epsilon = *ptr_epsilon;

  auto* ptr_start_n = reinterpret_cast<int*>(taskData->inputs[6]);
  start_n = *ptr_start_n;

  result = 0.0;
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[1] > 0u && taskData->inputs_count[2] > 0u;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::run() {
  internal_order_test();
  result = run_multistep_scheme_method_rectangle(f, f_values, lower_limits, upper_limits, dim, epsilon, start_n);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}