// Copyright 2024 Nesterov Alexander
#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

namespace prokhorov_n_global_search_algorithm_strongin_seq {

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  a = reinterpret_cast<double*>(taskData->inputs[0])[0];
  b = reinterpret_cast<double*>(taskData->inputs[1])[0];
  epsilon = reinterpret_cast<double*>(taskData->inputs[2])[0];

  f = [](double x) { return x * x; };

  result = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1 && taskData->inputs_count[2] == 1 &&
         taskData->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::run() {
  internal_order_test();

  result = stronginAlgorithm();
  std::this_thread::sleep_for(20ms);
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::stronginAlgorithm() {
  double x_min = a;
  double f_min = f(x_min);

  while ((b - a) > epsilon) {
    double x1 = a + (b - a) / 3.0;
    double x2 = b - (b - a) / 3.0;

    double f1 = f(x1);
    double f2 = f(x2);

    if (f1 < f2) {
      b = x2;
      if (f1 < f_min) {
        f_min = f1;
        x_min = x1;
      }
    } else {
      a = x1;
      if (f2 < f_min) {
        f_min = f2;
        x_min = x2;
      }
    }
  }

  return x_min;
}
}  // namespace prokhorov_n_global_search_algorithm_strongin_seq