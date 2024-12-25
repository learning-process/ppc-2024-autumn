#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->inputs_count[0] != 3) {
    return false;
  }

  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);

  lower_bound_ = inputs[0];
  upper_bound_ = inputs[1];
  n_ = static_cast<int>(inputs[2]);

  if (lower_bound_ >= upper_bound_) {
    return false;
  }

  if (n_ <= 0) {
    return false;
  }

  result_ = 0.0;

  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != 3) {
    return false;
  }

  if (taskData->outputs_count[0] != 1) {
    return false;
  }

  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);
  double lower_bound = inputs[0];
  double upper_bound = inputs[1];
  if (lower_bound >= upper_bound) {
    return false;
  }

  int n = static_cast<int>(inputs[2]);
  if (n <= 0) {
    return false;
  }

  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  result_ = integrate(f_, lower_bound_, upper_bound_, n_);
  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

void prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential::set_function(
    const std::function<double(double)>& func) {
  f_ = func;
}

double prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential::integrate(const std::function<double(double)>& f,
                                                                                 double lower_bound, double upper_bound,
                                                                                 int n) {
  double step = (upper_bound - lower_bound) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = lower_bound + (i + 0.5) * step;
    area += f(x) * step;
  }

  return area;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs_count[0] != 3) {
      return false;
    }

    auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);

    lower_bound_ = inputs[0];
    upper_bound_ = inputs[1];
    n_ = static_cast<int>(inputs[2]);

    if (lower_bound_ >= upper_bound_) {
      return false;
    }

    if (n_ <= 0) {
      return false;
    }

    result_ = 0.0;
  }

  boost::mpi::broadcast(world, lower_bound_, 0);
  boost::mpi::broadcast(world, upper_bound_, 0);
  boost::mpi::broadcast(world, n_, 0);

  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count[0] != 3) {
      return false;
    }

    if (taskData->outputs_count[0] != 1) {
      return false;
    }

    auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);
    double lower_bound = inputs[0];
    double upper_bound = inputs[1];
    if (lower_bound >= upper_bound) {
      return false;
    }

    int n = static_cast<int>(inputs[2]);
    if (n <= 0) {
      return false;
    }
  }

  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  result_ = parallel_integrate(f_, lower_bound_, upper_bound_, n_);
  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  }

  return true;
}

void prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel::set_function(
    const std::function<double(double)>& func) {
  f_ = func;
}

double prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel::parallel_integrate(
    const std::function<double(double)>& f, double lower_bound, double upper_bound, int n) {
  int rank = world.rank();
  int size = world.size();

  double step = (upper_bound - lower_bound) / n;
  double local_area = 0.0;
  int local_n = n / size;
  int start = rank * local_n;
  int end = (rank == size - 1) ? n : (rank + 1) * local_n;

  for (int i = start; i < end; ++i) {
    double x = lower_bound + (i + 0.5) * step;
    local_area += f(x) * step;
  }

  double global_area = 0.0;
  boost::mpi::reduce(world, local_area, global_area, std::plus<double>(), 0);

  return global_area;
}