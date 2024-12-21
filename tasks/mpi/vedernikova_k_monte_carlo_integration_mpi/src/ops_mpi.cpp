#include "mpi/vedernikova_k_monte_carlo_integration_mpi/include/ops_mpi.hpp"

#include <cstdio>
#include <functional>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/reduce.hpp"
#include "boost/mpi/communicator.hpp"

bool vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output (integration limits and result)
  a_x = *reinterpret_cast<double*>(taskData->inputs[0]);
  b_x = *reinterpret_cast<double*>(taskData->inputs[1]);
  a_y = *reinterpret_cast<double*>(taskData->inputs[2]);
  b_y = *reinterpret_cast<double*>(taskData->inputs[3]);
  a_z = *reinterpret_cast<double*>(taskData->inputs[4]);
  b_z = *reinterpret_cast<double*>(taskData->inputs[5]);
  num_points = *reinterpret_cast<int*>(taskData->inputs[6]);
  res = *reinterpret_cast<double*>(taskData->outputs[0]);
  return true;
}

bool vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 7 && taskData->outputs_count[0] == 1;
}

bool vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential::run() {
  internal_order_test();
  double x_limit = b_x - a_x;
  double y_limit = b_y - a_y;
  double z_limit = b_z - a_z;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(a_x, b_x);
  std::uniform_real_distribution<> dis_y(a_y, b_y);
  std::uniform_real_distribution<> dis_z(a_z, b_z);
  double sum = 0.0;
  for (int i = 0; i < num_points; i++) {
    double r = dis_x(gen);
    double theta = dis_y(gen);
    double phi = dis_z(gen);

    double x = r * std::sin(theta) * std::cos(phi);
    double y = r * std::sin(theta) * std::sin(phi);
    double z = r * std::cos(theta);
    double dV = r * r * std::sin(theta);
    sum += f(x, y, z) * dV;
  }
  res = x_limit * y_limit * z_limit * sum / num_points;
  return true;
}

bool vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    a_x = *reinterpret_cast<double*>(taskData->inputs[0]);
    b_x = *reinterpret_cast<double*>(taskData->inputs[1]);
    a_y = *reinterpret_cast<double*>(taskData->inputs[2]);
    b_y = *reinterpret_cast<double*>(taskData->inputs[3]);
    a_z = *reinterpret_cast<double*>(taskData->inputs[4]);
    b_z = *reinterpret_cast<double*>(taskData->inputs[5]);
    num_points = *reinterpret_cast<int*>(taskData->inputs[6]);

    res = *reinterpret_cast<double*>(taskData->outputs[0]);
  }
  return true;
}

bool vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs.size() == 7 && taskData->outputs.size() == 1 && *taskData->inputs[6] > 0;
  }
  return true;
}

bool vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, num_points, 0);
  boost::mpi::broadcast(world, a_x, 0);
  boost::mpi::broadcast(world, b_x, 0);
  boost::mpi::broadcast(world, a_y, 0);
  boost::mpi::broadcast(world, b_y, 0);
  boost::mpi::broadcast(world, a_z, 0);
  boost::mpi::broadcast(world, b_z, 0);

  double x_limit = b_x - a_x;
  double y_limit = b_y - a_y;
  double z_limit = b_z - a_z;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(a_x, b_x);
  std::uniform_real_distribution<> dis_y(a_y, b_y);
  std::uniform_real_distribution<> dis_z(a_z, b_z);
  double local_res = 0.0;
  int range = num_points / world.size();

  for (int i = 0; i < range; i++) {
    double r = dis_x(gen);
    double theta = dis_y(gen);
    double phi = dis_z(gen);

    double x = r * std::sin(theta) * std::cos(phi);
    double y = r * std::sin(theta) * std::sin(phi);
    double z = r * std::cos(theta);
    double dV = r * r * std::sin(theta);
    local_res += f(x, y, z) * dV;
  }

  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);

  if (world.rank() == 0) {
    res *= x_limit * y_limit * z_limit / num_points;
  }
  return true;
}
bool vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }

  return true;
}
