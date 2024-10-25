// Filatev Vladislav Sum_of_matrix_elements
#include "mpi/filatev_v_sum_of_matrix_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

std::vector<std::vector<int>> filatev_v_sum_of_matrix_elements_mpi::getRandomMatrix(int size_n, int size_m) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int>> matrix(size_m, std::vector<int>(size_n));

  for (int i = 0; i < size_m; ++i) {
    for (int j = 0;j < size_n; ++j){
      matrix[i][j] = gen() % 100;
    }
  }
  return matrix;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq::pre_processing() {
  internal_order_test();

  summ = 0;
  size_n = taskData->inputs_count[0];
  size_m = taskData->inputs_count[1];
  matrix = std::vector<int>(size_m * size_n);

  for (int i = 0; i < size_m; ++i) {
    auto* temp = reinterpret_cast<int*>(taskData->inputs[i]);

    for (int j = 0; j < size_n; ++j) {
      matrix[i * size_n + j] = temp[j];
    }
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

  for (int i = 0; i < matrix.size(); ++i) {
    summ += matrix[i];
  }

  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = summ;
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::pre_processing() {
  internal_order_test();
  int delta = 0, ras = 0;
  if (world.rank() == 0) {
    size_n = taskData->inputs_count[0];
    size_m = taskData->inputs_count[1];
    delta = (size_n * size_m) / world.size();
    ras = (size_n * size_m) % world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    matrix = std::vector<int>(size_m * size_n);
    for (int i = 0; i < size_m; ++i) {
      auto* temp = reinterpret_cast<int*>(taskData->inputs[i]);

      for (int j = 0; j < size_n; ++j) {
        matrix[i * size_n + j] = temp[j];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, matrix.data() + proc * delta + ras, delta);
    }
  }
  if (world.rank() == 0) {
    local_vector = std::vector<int>(matrix.begin(), matrix.begin() + delta + ras);
  } else {
    local_vector = std::vector<int>(delta);
    world.recv(0, 0, local_vector.data(), delta);
  }

  summ = 0;
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel::run() {
  internal_order_test();
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
