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
  std::cout << "marker8\n\n\n";
  internal_order_test();
  std::cout << "marker9\n\n\n";
  // Init value for input and output
    if (world.rank() == 0) {
    std::cout << "marker7\n\n\n";
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    std::cout << "marker5\n\n\n";
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    std::cout << "marker6\n\n\n";
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
  // local_res = f(local_a) + f(local_a + localPartsCount * h) * 0.5;

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par::validation()
{
  internal_order_test();
  std::cout << "marker10\n\n\n";
  // Check count elements of output
  if (world.rank() == 0) {
  std::cout << "m11 inputs_count ptr = " << taskData->inputs_count.data() << "\n\n\n";

std::cout << "m11 inputs_count = " << taskData->inputs_count[0] << "\n\n\n";
    uint32_t tmp1 = taskData->inputs_count[0];
    std::cout << "marker13\n\n\n";
    uint32_t tmp2 = taskData->outputs_count[0];
    std::cout << "marker14\n\n\n";
    return tmp1 == 3 && tmp2 == 1;
  }
  std::cout << "marker12\n\n\n";
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par::run()
{
  std::cout << "marker15\n\n\n";
  internal_order_test();
  std::cout << "marker16\n\n\n";

  double local_res = 0.0;
  local_res += f(local_a) + f(local_a + localPartsCount * h) * 0.5;

  for (unsigned int i = 0; i < localPartsCount; ++i) {
    double x0 = local_a + i * h;        // left trapezoid edge
    double x1 = local_a + (i + 1) * h;  // right trapezoid edge
    local_res += 0.5 * (f(x0) + f(x1)) * h; // trapezoid area
  }
  std::cout << "marker21\n\n\n";
  reduce(world, local_res, res, std::plus(), 0);
  std::cout << "marker22\n\n\n";
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