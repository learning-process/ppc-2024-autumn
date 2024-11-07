// Copyright 2023 Nesterov Alexander
#include "mpi/prokhorov_n_integral_rectangle_method/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <numeric>
#include <random>
#include <vector>


namespace prokhorov_n_integral_rectangle_method_mpi {

// Определение функции интегрирования для последовательного варианта
double TestMPITaskSequential::integrate(const std::function<double(double)>& f, double left_, double right_, int n) {
  double step = (right_ - left_) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = left_ + (i + 0.5) * step;
    area += f(x) * step;
  }

  return area;
}

// Определение функции set_function
void TestMPITaskSequential::set_function(const std::function<double(double)>& func) { func_ = func; }

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  uint8_t* inputs_raw = taskData->inputs[0];
  std::vector<double> inputs(reinterpret_cast<double*>(inputs_raw), reinterpret_cast<double*>(inputs_raw) + 3);
  left_ = inputs[0];
  right_ = inputs[1];
  n = static_cast<int>(inputs[2]);
  res = 0.0;
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  res = integrate(func_, left_, right_, n);
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

// Определение функции параллельного интегрирования с использованием MPI
double TestMPITaskParallel::parallel_integrate(const std::function<double(double)>& f, double left_, double right_,
                                               int n, const boost::mpi::communicator& world) {
  double range = right_ - left_;
  double step = range / n;

  int local_n = n / world.size();
  int start = world.rank() * local_n;
  int end = start + local_n;

  // Учитываем остаток для последнего процесса
  if (world.rank() == world.size() - 1) {
    end = n;
  }

  double local_result = 0.0;
  for (int i = start; i < end; ++i) {
    double x = left_ + (i + 0.5) * step;  // Центр каждого подотрезка
    local_result += f(x) * step;
  }

  double global_result;
  boost::mpi::reduce(world, local_result, global_result, std::plus<double>(), 0);

  // Раздаём результат всем процессам
  boost::mpi::broadcast(world, global_result, 0);

  return global_result;
}


// Определение функции set_function для параллельной задачи
void TestMPITaskParallel::set_function(const std::function<double(double)>& func) { func_ = func; }

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);
  left_ = inputs[0];
  right_ = inputs[1];
  n = static_cast<int>(inputs[2]);
  

  // Broadcasting variables to all processes
  boost::mpi::broadcast(world, left_, 0);
  boost::mpi::broadcast(world, right_, 0);
  boost::mpi::broadcast(world, n, 0);

  global_res = 0.0;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  local_res = parallel_integrate(func_, left_, right_, n, world);

  if (world.rank() == 0) {
    global_res = local_res;
  }
  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = global_res;
  }
  return true;
}

}  // namespace prokhorov_n_integral_rectangle_method_mpi
