// Copyright 2024 Sedova Olga
#include "seq/sedova_o_max_of_vector_elements/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;
std::vector<int> sedova_o_max_of_vector_elements_seq::generate_random_vector(int size, int value) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = random() % (value + 1);
  }
  return vec;
}

std::vector<std::vector<int>> sedova_o_max_of_vector_elements_seq::generate_random_matrix(int rows, int cols,
                                                                                          int value) {
  std::vector<std::vector<int>> matrix(rows);
  for (int i = 0; i < rows; i++) {
    matrix[i] = sedova_o_max_of_vector_elements_seq::generate_random_vector(cols, value);
  }
  return matrix;
}

int sedova_o_max_of_vector_elements_seq::find_max_of_matrix(const std::vector<int> matrix) {
  int max = matrix[0];
  for (int i = 0; i < matrix.size(); i++) {
    if (matrix[i] > max) {
      max = matrix[i];
    }
    return max;
 }

  bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::pre_processing() {
    internal_order_test();
    input_ = std::vector<int>(taskData->inputs_count[0] * taskData->inputs_count[1]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* input_data = reinterpret_cast<int*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
        input_[i * taskData->inputs_count[1] + j] = input_data[j];
      }
    }
    return true;
   }

  bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::validation() {
    internal_order_test();
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
  }

  bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::run() {
    internal_order_test();
    res_ = sedova_o_max_of_vector_elements_seq::find_max_of_matrix(input_);
    return true;
  }

  bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::post_processing() {
    internal_order_test();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
    return true;
  }

