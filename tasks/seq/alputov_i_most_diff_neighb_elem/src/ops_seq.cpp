// Copyright 2024 Alputov Ivan
#include "seq/alputov_i_most_diff_neighb_elem/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>

using namespace std::chrono_literals;

std::vector<int> alputov_i_most_diff_neighb_elem_seq::RandomVector(int sz) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-1000, 1000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = distrib(gen);
  }
  return vec;
}

int alputov_i_most_diff_neighb_elem_seq::SequentialTask::Max_Neighbour_Seq_Pos(const std::vector<int>& data) {
  if (data.size() < 2) {
    return 0;
  }
  int maxDifference = std::abs(data[0] - data[1]);
  int maxIndex = 0;
  for (size_t i = 1; i < data.size() - 1; ++i) {
    int difference = std::abs(data[i] - data[i + 1]);
    if (difference > maxDifference) {
      maxDifference = difference;
      maxIndex = i;
    }
  }
  return maxIndex;
}

bool alputov_i_most_diff_neighb_elem_seq::SequentialTask::pre_processing() {
  internal_order_test();
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
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result[0];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = result[1];
  return true;
}
