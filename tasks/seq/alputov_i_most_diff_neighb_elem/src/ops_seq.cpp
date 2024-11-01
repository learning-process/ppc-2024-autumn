// Copyright 2024 Alputov Ivan
#include "seq/alputov_i_most_diff_neighb_elem/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>

using namespace std::chrono_literals;

int alputov_i_most_diff_neighb_elem_seq::SequentialTask::Max_Neighbour_Seq_Pos(const std::vector<int>& data) {
  if (data.size() < 2) {
    return -1;
  }
  int maxDiff = std::abs(data[0] - data[1]);
  int maxIndex = 0;
  for (size_t i = 1; i < data.size() - 1; ++i) {
    int diff = std::abs(data[i] - data[i + 1]);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxIndex = i;
    }
  }
  return maxIndex;
}

bool alputov_i_most_diff_neighb_elem_seq::SequentialTask::pre_processing() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->inputs[0] == nullptr || taskData->inputs_count.empty()) {
    throw std::runtime_error("Input data is invalid.");
  }

  int size = taskData->inputs_count[0];
  if (size < 2) return false;
  inputData = std::vector<int>(size);
  memcpy(inputData.data(), taskData->inputs[0], sizeof(int) * size);
  return true;
}

bool alputov_i_most_diff_neighb_elem_seq::SequentialTask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2;
}

bool alputov_i_most_diff_neighb_elem_seq::SequentialTask::run() {
  internal_order_test();
  int index = Max_Neighbour_Seq_Pos(inputData);
  if (index == -1) {
    std::fill(result, result + 2, 0);
    return false;
  } else {
    result[0] = inputData[index];
    result[1] = inputData[index + 1];
    return true;
  }
}

bool alputov_i_most_diff_neighb_elem_seq::SequentialTask::post_processing() {
  internal_order_test();
  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr || taskData->outputs_count.empty() ||
      taskData->outputs_count[0] == 0) {
    throw std::runtime_error("Output data is invalid.");
  }
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result[0];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = result[1];
  return true;
}
