// Copyright 2024 Tarakanov Denis
#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

#include <thread>

using namespace std::chrono_literals;

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq::pre_processing()
{
  internal_order_test();

  // Init value for input and output
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  h = *reinterpret_cast<double*>(taskData->inputs[2]);
  res = 0;
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq::validation()
{
  internal_order_test();

  // Check count elements of output
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq::run()
{
    internal_order_test();

    int n = static_cast<int>((b - a) / h);
    double integral = 0.0;
    
    // summing trapezoid areas
    for (int i = 0; i < n; ++i) {
        double x0 = a + i * h;        // left trapezoid edge
        double x1 = a + (i + 1) * h;  // right trapezoid edge
        integral += 0.5 * (x0 * x0 + x1 * x1) * h; // trapezoid area
    }

    res = integral;

    return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq::post_processing()
{
  internal_order_test();

  *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par::pre_processing()
{
  internal_order_test();

  // Init value for input and output
    if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    h = *reinterpret_cast<double*>(taskData->inputs[2]);
    res = 0;
  }

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, h, 0);

  partsCount = (b - a) / h;
  localPartsCount = partsCount / world.size();
  localPartsCount = world.rank() < static_cast<int>(partsCount) % world.size() ? localPartsCount + 1 : localPartsCount;

  local_a = a + world.rank() * localPartsCount * h;
  local_res = f(local_a) + f(local_a + localPartsCount * h) * 0.5;

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par::validation()
{
  internal_order_test();

  // Check count elements of output
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par::run()
{
  internal_order_test();

  for (unsigned int i = 0; i < localPartsCount; ++i) {
    double x0 = local_a + i * h;        // left trapezoid edge
    double x1 = local_a + (i + 1) * h;  // right trapezoid edge
    local_res += 0.5 * (f(x0) + f(x1)) * h; // trapezoid area
  }
  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par::post_processing()
{
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}