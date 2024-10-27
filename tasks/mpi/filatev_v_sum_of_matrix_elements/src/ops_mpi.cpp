// Filatev Vladislav Sum_of_matrix_elements
#include "mpi/filatev_v_sum_of_matrix_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq::pre_processing() {
  internal_order_test();

  summ = 0;
  size_n = taskData->inputs_count[0];
  size_m = taskData->inputs_count[1];

  for (int i = 0; i < size_m; ++i) {
    auto* temp = reinterpret_cast<int*>(taskData->inputs[i]);

    matrix.insert(matrix.end(), temp, temp + size_n);
  }

  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq::run() {
  internal_order_test();

  summ = std::accumulate(matrix.begin(), matrix.end(), 0);

  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = summ;
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    size_n = taskData->inputs_count[0];
    size_m = taskData->inputs_count[1];

    for (int i = 0; i < size_m; ++i) {
      auto* temp = reinterpret_cast<int*>(taskData->inputs[i]);

      matrix.insert(matrix.end(), temp, temp + size_n);
    }
  }
  summ = 0;
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::run() {
  internal_order_test();
  int delta = 0;
  int ras = 0;

  if (world.rank() == 0 && world.size() > 1) {
    ras = (size_n * size_m) % (world.size() - 1);
    delta = (size_n * size_m) / (world.size() - 1);
  } else if (world.rank() == 0 && world.size() == 1) {
    ras = (size_n * size_m);
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 0; proc < (world.size() - 1); proc++) {
      world.send(proc + 1, 0, matrix.data() + proc * delta + ras, delta);
    }
    local_vector = std::vector<int>(matrix.begin(), matrix.begin() + ras);
  } else {
    local_vector = std::vector<int>(delta);
    world.recv(0, 0, local_vector.data(), delta);
  }
  long long local_summ = std::accumulate(local_vector.begin(), local_vector.end(), 0);
  reduce(world, local_summ, summ, std::plus(), 0);
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = summ;
  }
  return true;
}
