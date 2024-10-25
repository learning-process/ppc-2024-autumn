#include "seq/lysov_i_integration_the_trapezoid_method/include/ops_seq.hpp"

#include <thread>
using namespace std::chrono_literals;
bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.size() != 3) {
    return false;
  }
  if (taskData->outputs.size() != 1) {
    return false;
  }
  return true;
}

bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::pre_processing() {
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

bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::run() {
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

bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}