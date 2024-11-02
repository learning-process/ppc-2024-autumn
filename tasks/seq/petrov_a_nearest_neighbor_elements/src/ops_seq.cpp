// Copyright 2024 Nesterov Alexander
#include "seq/petrov_a_nearest_neighbor_elements/include/ops_seq.hpp"

#include <iostream>  // ��� ������ ���������� ����������
#include <cmath>
#include <limits>

using namespace std::chrono_literals;

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::pre_processing() {
 // std::cout << "Pre-processing started." << std::endl;
  
  internal_order_test();
  
  int size = taskData->inputs_count[0];
  input_.resize(size);

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  for (int i = 0; i < size; ++i) {
    input_[i] = input_data[i];
  }

  res.resize(2);  // ��� �������� ���� ��������� � ����������� ���������

 // std::cout << "Input data: ";
  //for (const auto& elem : input_) {
   // std::cout << elem << " ";
 // }
 // std::cout << std::endl;

  return true;
}

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::validation() {
  //std::cout << "Validation started." << std::endl;

  internal_order_test();
  
  bool isValid = (!taskData->inputs_count.empty()) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());
 // std::cout << "Validation result: " << (isValid ? "Passed" : "Failed") << std::endl;

  return isValid;
}

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::run() {
 // std::cout << "Run started." << std::endl;

  internal_order_test();
  
  size_t size = input_.size();
  if (size < 2) {
   // std::cout << "Not enough elements to find neighbors." << std::endl;
    return false;
  }

  // ����� ����������� �������� � �������� ���������
  int min_difference = std::numeric_limits<int>::max();
  size_t min_index = 0;

  for (size_t i = 0; i < size - 1; ++i) {
    int difference = std::abs(input_[i] - input_[i + 1]);
    if (difference < min_difference) {
      min_difference = difference;
      min_index = i;
    }
  }

  // ��������� ��������� �������� � res
  res[0] = input_[min_index];
  res[1] = input_[min_index + 1];

 // std::cout << "Minimum difference pair: " << res[0] << ", " << res[1] << std::endl;

  return true;
}

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::post_processing() {
 // std::cout << "Post-processing started." << std::endl;

  internal_order_test();
  
  // ���������� ��������� �������� � ����������� ��������� � �������� �����
  int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
  output_[0] = res[0];
  output_[1] = res[1];

 // std::cout << "Output data after post-processing: " << output_[0] << ", " << output_[1] << std::endl;

  return true;
}
