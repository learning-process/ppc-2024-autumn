#include "mpi/lysov_i_integration_the_trapezoid_method/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> lysov_i_integration_the_trapezoid_method_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.size() != 3) {
    std::cerr << "Error: Expected 3 inputs but got " << taskData->inputs.size() << std::endl;
    return false;
  }
  if (taskData->outputs.size() != 1) {
    std::cerr << "Error: Expected 1 output but got " << taskData->outputs.size() << std::endl;
    return false;
  }
  return true;
}

bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  cnt_of_splits = *reinterpret_cast<int*>(taskData->inputs[2]);
  h = (b - a) / cnt_of_splits;
  input_.resize(cnt_of_splits + 1);
  for (int i = 0; i <= cnt_of_splits; ++i) {
    double x = a + i * h;
    input_[i] = function(x);
  }
  return true;
}
bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  double result = 0.0;
  result += 0.5 * (function(a) + function(b));
  for (int i = 1; i < cnt_of_splits; ++i) {
    double x = a + i * h;
    result += function(x);
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
    if (taskData->inputs.size() != 3) {
      std::cerr << "Error:expected 3 inputs but got" << taskData->inputs.size() << std::endl;
      return false;
    }
    if (taskData->outputs.size() != 1) {
      std::cerr << "Error:expected 1 outputs but got" << taskData->outputs.size() << std::endl;
      return false;
    }
    cnt_of_splits = *reinterpret_cast<int*>(taskData->inputs[2]);
    if (cnt_of_splits <= 0) {
      std::cerr << "Error: count of splits must be greather than 0" << std::endl;
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
    cnt_of_splits = *reinterpret_cast<int*>(taskData->inputs[2]);
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
  local_res += 0.5 * (function(local_a) + function(local_a + local_cnt_of_splits * h));
  for (int i = 0; i < local_cnt_of_splits; i++) {
    double x = local_a + i * h;
    local_res += function(x);
  }
  local_res *= h;
  boost::mpi::reduce(world, local_res, res, std::plus<double>(), 0);  // ?
  return true;
}

bool lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  // double global_result = 0.0;
  // boost::mpi::reduce(world, res, global_result, std::plus<double>(), 0);// ?
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}
