#include "mpi/plekhanov_d_trapez_integration/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <string>
#include <vector>

bool plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ::pre_processing() {
  internal_order_test();
  a_ = *reinterpret_cast<double *>(taskData->inputs[0]);
  b_ = *reinterpret_cast<double *>(taskData->inputs[1]);
  n_ = *reinterpret_cast<int *>(taskData->inputs[2]);
  return true;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ::run() {
  internal_order_test();
  res_ = integrate_function(a_, b_, n_, function_);
  return true;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ::post_processing() {
  internal_order_test();
  *reinterpret_cast<double *>(taskData->outputs[0]) = res_;
  return true;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    a_ = *reinterpret_cast<double *>(taskData->inputs[0]);
    b_ = *reinterpret_cast<double *>(taskData->inputs[1]);
    n_ = *reinterpret_cast<int *>(taskData->inputs[2]);
  }
  return true;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI::run() {
  internal_order_test();
  double params[3] = {0.0};
  if (world.rank() == 0) {
    params[0] = a_;
    params[1] = b_;
    params[2] = n_;
  }
  boost::mpi::broadcast(world, params, std::size(params), 0);
  double local_res = integrate_function(params[0], params[1], static_cast<int>(params[2]), function_);
  boost::mpi::reduce(world, local_res, res_, std::plus(), 0);
  return true;
}
bool plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double *>(taskData->outputs[0]) = res_;
  }
  return true;
}
void plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ::set_function(const std::function<double(double)> &f) {
  function_ = f;
}
void plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI::set_function(const std::function<double(double)> &f) {
  function_ = f;
}
double plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ::integrate_function(
    double a, double b, int n, const std::function<double(double)> &f) {
  const double width = (b - a) / n;
  double result = 0.0;
  for (int step = 0; step < n; step++) {
    const double x1 = a + step * width;
    const double x2 = a + (step + 1) * width;
    result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
  }
  return result;
}
double plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI::integrate_function(
    double a, double b, int n, const std::function<double(double)> &f) {
  int rank = world.rank();
  int size = world.size();
  const double width = (b - a) / n;
  double result = 0.0;
  for (int step = rank; step < n; step += size) {
    const double x1 = a + step * width;
    const double x2 = a + (step + 1) * width;
    result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
  }
  return result;
}