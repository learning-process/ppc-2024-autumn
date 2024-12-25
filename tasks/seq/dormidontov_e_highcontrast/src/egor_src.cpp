#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "seq/dormidontov_e_highcontrast/include/egor_include.hpp"

namespace dormidontov_e_highcontrast_seq {
// use to control diap of data
inline void clip(int& x) {
  if (x > 255) x = 255;
  if (x < 0) x = 0;
}
}  // namespace dormidontov_e_highcontrast_seq

bool dormidontov_e_highcontrast_seq::ContrastS::pre_processing() {
  internal_order_test();
  size = taskData->inputs_count[0];
  y.resize(size);
  res_.resize(size);
  auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(ptr, ptr + size, y.data());
  return true;
}

bool dormidontov_e_highcontrast_seq::ContrastS::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 1 && taskData->inputs_count[0] > 0) &&
         (taskData->outputs.size() == 1 && taskData->outputs_count[0] > 0);
}

bool dormidontov_e_highcontrast_seq::ContrastS::run() {
  internal_order_test();
  ymin = 255;
  ymax = 0;
  for (int i = 0; i < size; ++i) {
    ymin = std::min(y[i], ymin);
    ymax = std::max(y[i], ymax);
  }
  clip(ymin);
  clip(ymax);
  for (int i = 0; i < size; ++i) {
    res_[i] = ((y[i] - ymin) * 255) / (ymax - ymin);
    clip(res_[i]);
  }
  return true;
}

bool dormidontov_e_highcontrast_seq::ContrastS::post_processing() {
  internal_order_test();
  auto* tmp = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.data(), res_.data() + size, tmp);
  return true;
}