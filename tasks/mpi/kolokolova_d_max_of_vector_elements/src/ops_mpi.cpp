// Copyright 2023 Nesterov Alexander
#include "mpi/kolokolova_d_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

//std::vector<std::vector<int>> kolokolova_d_max_of_vector_elements_mpi::getRandomVector(int sz) {
//  std::random_device dev;
//  std::mt19937 gen(dev());
//  std::vector<std::vector<int>> vec(sz, std::vector<int>(sz));
//  for (int i = 0; i < sz; i++) {
//    for (int j = 0; j < sz; j++) {
//      vec[i][j] = gen() % 100;
//    }  
//  }
//  return vec;
//}
bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_t row_count = static_cast<size_t>(*taskData->inputs[1]);
  size_t col_count = taskData->inputs_count[0] / row_count;

  input_.resize(row_count, std::vector<int>(col_count));

  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < row_count; ++i) {
    for (size_t j = 0; j < col_count; ++j) {
      input_[i][j] = input_ptr[i * col_count + j];
    }
  }
  res.resize(row_count);
  return true;
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return *taskData->inputs[1] == taskData->outputs_count[0];
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); ++i) {
    int max_value = input_[i][0];
    for (size_t j = 1; j < input_[i].size(); ++j) {
      if (input_[i][j] > max_value) {
        max_value = input_[i][j];
      }
    }
    res[i] = max_value;
  }
  return true;
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_ptr[i] = res[i];
    //std::cout << "Output Max[" << i << "]: " << res[i] << std::endl;  // Для отладки
  }
  return true;
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  size_t row_count = static_cast<size_t>(*taskData->inputs[1]);
  std::cout << row_count << "\n";
  int ProcNum = world.size();
  int ProcRank = world.rank();
  int N = (taskData->inputs_count[0]);
  std::cout << N << "\n";

  if (ProcRank == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }

  int size = N / ProcNum;
  int remainder = N % ProcNum;
  if (ProcRank < remainder) {
    size++;
  }
  local_input_.resize(size);
  res.resize(row_count);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), size, 0);
  std::cout << "ProcRang: " << ProcRank << " Nums: " << local_input_.data() << "\n";
  return true;
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[0] != 0;
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int ProcRank = world.rank();
  max = std::numeric_limits<int>::min();  // Для корректной работы с минимальным значением
  for (size_t i = 0; i < local_input_.size(); i++) {
    if (max < local_input_[i]) max = local_input_[i];
    std::cout << "ProcRang: " << ProcRank << " Nums: " << local_input_[i] << "\n";
  }
  return true;
}

bool kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  boost::mpi::gather(world, max, res.data(), 0);
  if (world.rank() == 0) {
    std::memcpy(reinterpret_cast<int*>(taskData->outputs.data()), res.data(), res.size() * sizeof(int));
  }

  /*if (world.rank() == 0) {
    for (size_t i = 0; i < res.size(); ++i) {
      reinterpret_cast<int*>(taskData->outputs.data())[i] = res[i];
    }
  }*/

  return true;
}