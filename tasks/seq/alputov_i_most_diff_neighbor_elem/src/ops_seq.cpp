// Copyright 2024 Alputov Ivan
#include "seq/alputov_i_most_diff_neighbor_elem/include/ops_seq.hpp"

#include <algorithm>  
#include <stdexcept>  
#include <thread>

using namespace std::chrono_literals;

namespace alputov_i_most_diff_neighbor_elem_seq {


std::pair<int, int> MostDiffNeighborElemSeq::findMaxDifferencePair(const std::vector<int>& vec) {
  if (vec.empty()) {
    return {0, 0};  
  }

  std::pair<int, int> maxPair = {0, 0};
  int maxDifference = 0;

  if (vec.size() > 1) {
    maxPair = {vec[0], vec[1]};
    maxDifference = std::abs(vec[1] - vec[0]);
  }

  for (size_t i = 1; i < vec.size(); ++i) {
    int currentDifference = std::abs(vec[i] - vec[i - 1]);
    if (currentDifference >= maxDifference) {
      maxDifference = currentDifference;
      maxPair = {vec[i - 1], vec[i]};
    }
  }
  return maxPair;
}

bool MostDiffNeighborElemSeq::pre_processing() {
  internal_order_test();

   if (taskData->inputs.empty() || taskData->inputs[0] == nullptr || taskData->inputs_count.empty() ||
      taskData->inputs_count[0] == 0) {
    throw std::runtime_error("Input data is invalid.");
  }

  int* inputPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  size_t inputSize = taskData->inputs_count[0];
  inputVec_.resize(inputSize);
  std::copy(inputPtr, inputPtr + inputSize, inputVec_.begin());

  maxDifferencePair_ = {0, 0};
  return true;
}

bool MostDiffNeighborElemSeq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 1;
}

bool MostDiffNeighborElemSeq::run() {
  internal_order_test();
  maxDifferencePair_ = findMaxDifferencePair(inputVec_);
  return true;
}

bool MostDiffNeighborElemSeq::post_processing() {
  internal_order_test();

  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr || taskData->outputs_count.empty() ||
      taskData->outputs_count[0] == 0) {
    throw std::runtime_error("Output data is invalid.");
  }

  std::pair<int, int>* outputPtr = reinterpret_cast<std::pair<int, int>*>(taskData->outputs[0]);
  *outputPtr = maxDifferencePair_;

  return true;
}

}  // namespace alputov_i_most_diff_neighbor_elem_seq