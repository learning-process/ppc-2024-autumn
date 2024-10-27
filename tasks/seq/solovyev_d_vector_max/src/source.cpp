#include <random>
#include <thread>

#include "seq/solovyev_d_vector_max/include/header.hpp"

using namespace std::chrono_literals;

int solovyev_d_vector_max_mpi::vectorMax(std::vector<int, std::allocator<int>> v) {
  int m = -214748364;
  for (std::string::size_type i = 0; i < v.size(); i++) {
    if (v[i] > m) {
      m = v[i];
    }
  }
  return m;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::pre_processing() {
  internal_order_test();

  // Init data vector
  int* input_ = reinterpret_cast<int*>(taskData->inputs[0]);
  data = std::vector<int>(input_, input_ + taskData->inputs_count[0]);
  // Init result value
  result = 0;
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::run() {
  internal_order_test();

  // Determine maximum value of data vector
  result = vectorMax(data);
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}