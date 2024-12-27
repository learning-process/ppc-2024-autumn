#include "seq/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <thread>
#include <vector>

bool kurakin_m_graham_scan_seq::isLeftAngle(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3) {
  return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < 0;
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  count_point = (int)taskData->inputs_count[0] / 2;
  input_ = std::vector<std::vector<double>>(count_point, std::vector<double>(2, 0));
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < count_point * 2; i += 2) {
    input_[i / 2][0] = tmp_ptr[i];
    input_[i / 2][1] = tmp_ptr[i + 1];
  }

  return true;
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 1 && taskData->outputs.size() == 2 &&
         taskData->outputs_count.size() == 2 && taskData->inputs_count[0] % 2 == 0 &&
         taskData->inputs_count[0] / 2 > 2 && taskData->outputs_count[0] == 1 &&
         taskData->inputs_count[0] == taskData->outputs_count[1];
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::run() {
  internal_order_test();

  int ind_min_y = std::min_element(input_.begin(), input_.end(),
                                   [&](std::vector<double> a, std::vector<double> b) {
                                     return a[1] < b[1] || (a[1] == b[1] && a[0] > b[0]);
                                   }) -
                  input_.begin();
  std::swap(input_[0], input_[ind_min_y]);
  std::sort(input_.begin() + 1, input_.end(),
            [&](std::vector<double> a, std::vector<double> b) { return isLeftAngle(a, input_[0], b); });

  int k = 1;
  for (int i = 2; i < count_point; i++) {
    while (k > 0 && isLeftAngle(input_[k - 1], input_[k], input_[i])) {
      k--;
    }
    std::swap(input_[i], input_[k + 1]);
    k++;
  }
  count_point = k + 1;

  return true;
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = count_point;
  for (int i = 0; i < count_point * 2; i += 2) {
    reinterpret_cast<double*>(taskData->outputs[1])[i] = input_[i / 2][0];
    reinterpret_cast<double*>(taskData->outputs[1])[i + 1] = input_[i / 2][1];
  }

  return true;
}
