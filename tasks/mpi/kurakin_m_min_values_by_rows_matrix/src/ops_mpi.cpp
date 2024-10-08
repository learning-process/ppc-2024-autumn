#include "mpi/kurakin_m_min_values_by_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = std::vector<int>(1, 0);
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  
  for (int i = 0; i < count_rows; i++) {
    int min_el = INT_MAX;
    for (int j = 0; j < size_rows; j++) {
      min_el = std::min(min_el, input_[j + i * size_rows]);  
      //res[i] = *std::min_element(input_.begin() + i * size_rows, input_.begin() + (i + 1) * size_rows);
    }
    res[i] = min_el;
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    reinterpret_cast<int*>(taskData->outputs[i])[0] = res[i];
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta_rows = 0;
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta_rows = taskData->inputs_count[0] / world.size() / count_rows;
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    int ind = 0;
    for (unsigned rank = 0; rank < (unsigned)world.size(); rank++) {
      for (unsigned rows = 0; rows < (unsigned)count_rows; rows++) {
        for (unsigned i = 0; i < delta_rows; i++) {
          input_[ind] = tmp_ptr[i + rows * size_rows + rank * delta_rows];
          ind++;
        }
      }
    }

    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = std::vector<int>(count_rows, 0);
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int delta = local_input_.size() / count_rows;
  for (int i = 0; i < count_rows; i++) {
    int local_res;
    local_res = *std::min_element(local_input_.begin() + i * delta, local_input_.begin() + (i + 1) * delta);
    world.barrier();
    reduce(world, local_res, res[i], boost::mpi::minimum<int>(), 0);
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < count_rows; i++) {
      reinterpret_cast<int*>(taskData->outputs[i])[0] = res[i];
    }
  }
  return true;
}
