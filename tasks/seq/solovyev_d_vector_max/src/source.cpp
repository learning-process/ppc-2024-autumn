#include <random>
#include <thread>

#include "seq/solovyev_d_vector_max/include/header.hpp"

using namespace std::chrono_literals;

std::vector<int> solovyev_d_vector_max_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

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
  data = std::vector<int>(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    data[i] = tempPtr[i];
  }

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
  result = solovyev_d_vector_max_mpi::vectorMax(data);
  return true;
}

bool solovyev_d_vector_max_mpi::VectorMaxSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}