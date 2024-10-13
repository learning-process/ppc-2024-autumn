// Copyright 2024 Nesterov Alexander
#include "seq/korovin_n_min_val_row_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool korovin_n_min_val_row_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];

  input_.resize(rows, std::vector<int>(cols));

  for(size_t i = 0; i < rows; i++){
    int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
    for(size_t j = 0; j < cols; j++){
      input_[i][j] = input_matrix[j]; 
    }
  }
  res.resize(rows);
  return true;
}

bool korovin_n_min_val_row_matrix_seq::TestTaskSequential::validation() {
  internal_order_test(); 

  if (taskData->inputs.empty() || taskData->outputs.empty()){
    return false;
  }
  if (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0){
    return false;
  }
  if (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != taskData->inputs_count[0]){
    return false;
  }
  return true;
}

bool korovin_n_min_val_row_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for(size_t i = 0; i < input_.size(); i++){
    int min_val = input_[i][0];
    for(size_t j = 1; j < input_[i].size(); j++){
      if (input_[i][j] < min_val){
        min_val = input_[i][j];
      }
    }
    res[i] = min_val;
  }
  return true;
}

bool korovin_n_min_val_row_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
  for(size_t i = 0; i < res.size(); i++){
    output_matrix[i] = res[i];
  }
  return true;
}

std::vector<int> korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_vector(int size, int lower_bound, int upper_bound) {
  std::vector<int> v1(size);
  for(auto& num: v1){
    num = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return v1;
}

std::vector<std::vector<int>> korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(int rows, int cols) {
  std::vector<std::vector<int>> matrix1(rows, std::vector<int>(cols));
  for(auto& row: matrix1){
    row = generate_rnd_vector(cols, -25, 50); 
    int rnd_index = std::rand() % cols;
    row[rnd_index] = -25; 
  }
  return matrix1;
}