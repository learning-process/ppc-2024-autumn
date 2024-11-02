#include "seq/lysov_i_integration_the_trapezoid_method/include/ops_seq.hpp"

#include <thread>
using namespace std::chrono_literals;
bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::pre_processing() {
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

bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::run() {
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

bool lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}