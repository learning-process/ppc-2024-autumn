#include "seq/vasilev_s_nearest_neighbor_elements/include/ops_seq.hpp"

#include <cmath>
#include <limits>
#include <thread>

using namespace std::chrono_literals;

bool vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

  min_diff_ = std::numeric_limits<int>::max();
  index1_ = -1;
  index2_ = -1;
  return true;
}

bool vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential::validation() {
  internal_order_test();
  return !taskData->inputs_count.empty() && taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 3;
}

bool vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size() - 1; ++i) {
    int diff = std::abs(input_[i + 1] - input_[i]);
    if (diff < min_diff_) {
      min_diff_ = diff;
      index1_ = static_cast<int>(i);
      index2_ = static_cast<int>(i + 1);
    }
  }
  return true;
}

bool vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  output_ptr[0] = min_diff_;
  output_ptr[1] = index1_;
  output_ptr[2] = index2_;
  return true;
}
