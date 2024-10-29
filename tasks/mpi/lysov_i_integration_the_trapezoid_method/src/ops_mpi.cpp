#include "mpi/lysov_i_integration_the_trapezoid_method/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  cnt_of_splits = static_cast<int>(std::abs((b - a)) / epsilon);
  h = (b - a) / cnt_of_splits;
  input_.resize(cnt_of_splits + 1);
  for (int i = 0; i <= cnt_of_splits; ++i) {
    double x = a + i * h;
    input_[i] = function_square(x);
  }
  return true;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  double result = 0.0;
  result += 0.5 * (function_square(a) + function_square(b));
  for (int i = 1; i < cnt_of_splits; ++i) {
    double x = a + i * h;
    result += function_square(x);
  }
  result *= h;
  res = result;
  return true;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs.size() != 3) || (taskData->outputs.size() != 1)) {
      return false;
    }
    double epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
    if (epsilon <= 0) {
      return false;
    }
  }
  return true;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    double epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
    cnt_of_splits = static_cast<int>(std::abs((b - a)) / epsilon);
  }
  
  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, cnt_of_splits, 0);

  h = (b - a) / cnt_of_splits;
  local_cnt_of_splits = cnt_of_splits / world.size();
  if (world.rank() < cnt_of_splits % world.size()) {
    local_cnt_of_splits++;
  }
  local_a = a + world.rank() * local_cnt_of_splits * h;
  local_input_.resize(local_cnt_of_splits + 1);
  return true;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  double local_res = 0.0;
  local_res += 0.5 * (function_square(local_a) + function_square(local_a + local_cnt_of_splits * h));
  for (int i = 0; i < local_cnt_of_splits; i++) {
    double x = local_a + i * h;
    local_res += function_square(x);
  }
  local_res *= h;
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}
