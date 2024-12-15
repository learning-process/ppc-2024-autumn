#include "mpi/bessonov_e_integration_monte_carlo/include/ops_mpi.hpp"

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  num_points = *reinterpret_cast<int*>(taskData->inputs[2]);
  return true;
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);

  double sum = 0.0;
  for (int i = 0; i < num_points; ++i) {
    double x = dis(gen);
    sum += exampl_func(x);
  }
  res = (b - a) * (sum / num_points);
  return true;
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs.size() != 3) || (taskData->outputs.size() != 1)) {
      return false;
    }
    num_points = *reinterpret_cast<int*>(taskData->inputs[2]);
    if (num_points <= 0) {
      return false;
    }
  }
  return true;
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    num_points = *reinterpret_cast<int*>(taskData->inputs[2]);
  }

  return true;
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, num_points, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);

  int remainder = num_points % world.size();
  int num_points_for_process = num_points / world.size() + (world.rank() < remainder ? 1 : 0);

  double sum = 0.0;
  for (int i = 0; i < num_points_for_process; ++i) {
    double x = dis(gen);
    sum += exampl_func(x);
  }

  boost::mpi::reduce(world, sum, res, std::plus<>(), 0);
  if (world.rank() == 0) {
    res = (b - a) * res / num_points;
  }
  return true;
}

bool bessonov_e_integration_monte_carlo_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}
