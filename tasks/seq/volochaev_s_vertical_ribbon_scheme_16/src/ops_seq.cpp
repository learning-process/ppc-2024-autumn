#include "seq/volochaev_s_vertical_ribbon_scheme_16/include/ops_seq.hpp"

#include <functional>
#include <random>
#include <thread>

using namespace std::chrono_literals;

bool volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16::pre_processing() {
  internal_order_test();

  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);
  int c = taskData->inputs_count[0];
  m = taskData->inputs_count[1];
  n = c / m;
  res.assign(n, 0);

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16::run() {
  internal_order_test();

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res[j] += input_A[i * n + j] * input_B[i];
    }
  }

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16::post_processing() {
  internal_order_test();

  int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), ans);

  return true;
}
