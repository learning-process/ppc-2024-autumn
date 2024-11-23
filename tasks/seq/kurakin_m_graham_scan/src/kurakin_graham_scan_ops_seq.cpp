#include "seq/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>
#include <cmath>

using namespace std::chrono_literals;

bool kurakin_m_graham_scan_seq::isLeftAngle(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3) {
  return ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2])) < 0;
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 2 && taskData->inputs_count.size() == 2 && taskData->outputs.size() == 3 &&
         taskData->outputs_count.size() == 3 && taskData->inputs_count[0] > 2 &&
         taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1 &&
         taskData->outputs_count[1] == taskData->outputs_count[2] &&
         taskData->inputs_count[0] == taskData->outputs_count[1];
}

bool kurakin_m_graham_scan_seq::TestTaskSequential::run() {
  internal_order_test();
  count_point = (int)taskData->inputs_count[0];
  input_ = std::vector<std::vector<double>>(count_point, std::vector<double>(3, 0));
  auto* x_tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* y_tmp_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < count_point; i++) {
    input_[i][1] = x_tmp_ptr[i];
    input_[i][2] = y_tmp_ptr[i];
  }

  int ind_min_y =
      std::min_element(input_.begin(), input_.end(),
                                   [&](std::vector<double> a, std::vector<double> b) {
                                     return a[2] < b[2] || (a[2] == b[2] && a[1] > b[1]);
                                   }) -
                  input_.begin();
  std::swap(input_[0], input_[ind_min_y]);
  for (int i = 1; i < count_point; i++) {
    input_[i][0] =
        (input_[i][1] - input_[0][1]) / sqrt(pow(input_[i][1] - input_[0][1], 2) + pow(input_[i][2] - input_[0][2], 2));
  }
  std::sort(input_.begin() + 1, input_.end(), [&](std::vector<double> a, std::vector<double> b) {
    return a[0] > b[0];                          
  });

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
  for (int i = 0; i < count_point; i++) {
    reinterpret_cast<double*>(taskData->outputs[1])[i] = input_[i][1];
    reinterpret_cast<double*>(taskData->outputs[2])[i] = input_[i][2];
  }
  return true;
}
