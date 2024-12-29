#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

namespace nasedkin_e_strassen_algorithm {

bool StrassenAlgorithmSEQ::pre_processing() {
  internal_order_test();
  auto* inputsA = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* inputsB = reinterpret_cast<double*>(taskData->inputs[1]);
  matrixSize = static_cast<size_t>(std::sqrt(taskData->inputs_count[0]));
  size_t newSize = nextPowerOfTwo(matrixSize);
  inputMatrixA = padMatrix(std::vector<double>(inputsA, inputsA + matrixSize * matrixSize), matrixSize, newSize);
  inputMatrixB = padMatrix(std::vector<double>(inputsB, inputsB + matrixSize * matrixSize), matrixSize, newSize);
  outputMatrix.resize(newSize * newSize, 0.0);
  matrixSize = newSize;
  return true;
}

bool StrassenAlgorithmSEQ::validation() {
  internal_order_test();
  if (taskData->inputs.empty()) {
    return false;
  }
  if (taskData->inputs_count[0] != taskData->inputs_count[1]) {
    return false;
  }
  if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool StrassenAlgorithmSEQ::run() {
  internal_order_test();
  outputMatrix = strassen_multiply_seq(inputMatrixA, inputMatrixB, matrixSize);
  return true;
}

bool StrassenAlgorithmSEQ::post_processing() {
  internal_order_test();
  auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
  auto originalSize = static_cast<size_t>(std::sqrt(taskData->outputs_count[0]));
  for (size_t i = 0; i < originalSize; ++i) {
    for (size_t j = 0; j < originalSize; ++j) {
      outputs[i * originalSize + j] = outputMatrix[i * matrixSize + j];
    }
  }
  return true;
}

bool StrassenAlgorithmMPI::pre_processing() {
  internal_order_test();
  int rank = world.rank();
  if (rank == 0) {
    auto* inputsA = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* inputsB = reinterpret_cast<double*>(taskData->inputs[1]);

    if (inputsA == nullptr || inputsB == nullptr) {
      return false;
    }

    matrixSize = static_cast<size_t>(std::sqrt(taskData->inputs_count[0]));

    if (matrixSize * matrixSize != taskData->inputs_count[0]) {
      return false;
    }

    size_t newSize = nextPowerOfTwo(matrixSize);
    inputMatrixA = padMatrix(std::vector<double>(inputsA, inputsA + matrixSize * matrixSize), matrixSize, newSize);
    inputMatrixB = padMatrix(std::vector<double>(inputsB, inputsB + matrixSize * matrixSize), matrixSize, newSize);
    outputMatrix.resize(newSize * newSize, 0.0);
    matrixSize = newSize;
  }

  boost::mpi::broadcast(world, matrixSize, 0);
  if (rank != 0) {
    inputMatrixA.resize(matrixSize * matrixSize);
    inputMatrixB.resize(matrixSize * matrixSize);
    outputMatrix.resize(matrixSize * matrixSize, 0.0);
  }
  boost::mpi::broadcast(world, inputMatrixA, 0);
  boost::mpi::broadcast(world, inputMatrixB, 0);

  return true;
}

bool StrassenAlgorithmMPI::validation() {
  internal_order_test();
  int rank = world.rank();
  if (rank == 0) {
    if (taskData->inputs.empty()) {
      return false;
    }
    if (taskData->inputs_count[0] != taskData->inputs_count[1]) {
      return false;
    }
    if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool StrassenAlgorithmMPI::run() {
  internal_order_test();
  outputMatrix = strassen_multiply(inputMatrixA, inputMatrixB, matrixSize);
  return true;
}

bool StrassenAlgorithmMPI::post_processing() {
  internal_order_test();
  int rank = world.rank();
  if (rank == 0) {
    auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
    auto originalSize = static_cast<size_t>(std::sqrt(taskData->outputs_count[0]));
    for (size_t i = 0; i < originalSize; ++i) {
      for (size_t j = 0; j < originalSize; ++j) {
        outputs[i * originalSize + j] = outputMatrix[i * matrixSize + j];
      }
    }
  }
  return true;
}

std::vector<double> matrix_add(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size) {
  std::vector<double> result(size * size);
  for (size_t i = 0; i < size * size; ++i) {
    result[i] = matrixA[i] + matrixB[i];
  }
  return result;
}

std::vector<double> matrix_subtract(const std::vector<double>& matrixA, const std::vector<double>& matrixB,
                                    size_t size) {
  std::vector<double> result(size * size);
  for (size_t i = 0; i < size * size; ++i) {
    result[i] = matrixA[i] - matrixB[i];
  }
  return result;
}

size_t nextPowerOfTwo(size_t n) {
  size_t power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
}

std::vector<double> padMatrix(const std::vector<double>& matrix, size_t originalSize, size_t newSize) {
  std::vector<double> padded(newSize * newSize, 0.0);
  for (size_t i = 0; i < originalSize; ++i) {
    for (size_t j = 0; j < originalSize; ++j) {
      padded[i * newSize + j] = matrix[i * originalSize + j];
    }
  }
  return padded;
}

std::vector<double> strassen_recursive(const std::vector<double>& matrixA, const std::vector<double>& matrixB,
                                       size_t size) {
  if (size == 1) {
    return {matrixA[0] * matrixB[0]};
  }

  size_t half_size = size / 2;

  std::vector<double> A11(half_size * half_size);
  std::vector<double> A12(half_size * half_size);
  std::vector<double> A21(half_size * half_size);
  std::vector<double> A22(half_size * half_size);

  std::vector<double> B11(half_size * half_size);
  std::vector<double> B12(half_size * half_size);
  std::vector<double> B21(half_size * half_size);
  std::vector<double> B22(half_size * half_size);

  for (size_t i = 0; i < half_size; ++i) {
    for (size_t j = 0; j < half_size; ++j) {
      A11[i * half_size + j] = matrixA[i * size + j];
      A12[i * half_size + j] = matrixA[i * size + j + half_size];
      A21[i * half_size + j] = matrixA[(i + half_size) * size + j];
      A22[i * half_size + j] = matrixA[(i + half_size) * size + j + half_size];

      B11[i * half_size + j] = matrixB[i * size + j];
      B12[i * half_size + j] = matrixB[i * size + j + half_size];
      B21[i * half_size + j] = matrixB[(i + half_size) * size + j];
      B22[i * half_size + j] = matrixB[(i + half_size) * size + j + half_size];
    }
  }

  std::vector<double> M1 =
      strassen_recursive(matrix_add(A11, A22, half_size), matrix_add(B11, B22, half_size), half_size);
  std::vector<double> M2 = strassen_recursive(matrix_add(A21, A22, half_size), B11, half_size);
  std::vector<double> M3 = strassen_recursive(A11, matrix_subtract(B12, B22, half_size), half_size);
  std::vector<double> M4 = strassen_recursive(A22, matrix_subtract(B21, B11, half_size), half_size);
  std::vector<double> M5 = strassen_recursive(matrix_add(A11, A12, half_size), B22, half_size);
  std::vector<double> M6 =
      strassen_recursive(matrix_subtract(A21, A11, half_size), matrix_add(B11, B12, half_size), half_size);
  std::vector<double> M7 =
      strassen_recursive(matrix_subtract(A12, A22, half_size), matrix_add(B21, B22, half_size), half_size);

  std::vector<double> C11 = matrix_add(matrix_subtract(matrix_add(M1, M4, half_size), M5, half_size), M7, half_size);
  std::vector<double> C12 = matrix_add(M3, M5, half_size);
  std::vector<double> C21 = matrix_add(M2, M4, half_size);
  std::vector<double> C22 = matrix_add(matrix_subtract(matrix_add(M1, M3, half_size), M2, half_size), M6, half_size);

  std::vector<double> local_result(size * size);
  for (size_t i = 0; i < half_size; ++i) {
    for (size_t j = 0; j < half_size; ++j) {
      local_result[i * size + j] = C11[i * half_size + j];
      local_result[i * size + j + half_size] = C12[i * half_size + j];
      local_result[(i + half_size) * size + j] = C21[i * half_size + j];
      local_result[(i + half_size) * size + j + half_size] = C22[i * half_size + j];
    }
  }

  return local_result;
}

std::vector<double> StrassenAlgorithmSEQ::strassen_multiply_seq(const std::vector<double>& matrixA,
                                                                const std::vector<double>& matrixB, size_t size) {
  if (size == 1) {
    return {matrixA[0] * matrixB[0]};
  }

  size_t new_size = 1;
  while (new_size < size) {
    new_size *= 2;
  }

  size_t half_size = new_size / 2;

  std::vector<double> A11(half_size * half_size);
  std::vector<double> A12(half_size * half_size);
  std::vector<double> A21(half_size * half_size);
  std::vector<double> A22(half_size * half_size);

  std::vector<double> B11(half_size * half_size);
  std::vector<double> B12(half_size * half_size);
  std::vector<double> B21(half_size * half_size);
  std::vector<double> B22(half_size * half_size);

  for (size_t i = 0; i < half_size; ++i) {
    for (size_t j = 0; j < half_size; ++j) {
      A11[i * half_size + j] = matrixA[i * new_size + j];
      A12[i * half_size + j] = matrixA[i * new_size + j + half_size];
      A21[i * half_size + j] = matrixA[(i + half_size) * new_size + j];
      A22[i * half_size + j] = matrixA[(i + half_size) * new_size + j + half_size];

      B11[i * half_size + j] = matrixB[i * new_size + j];
      B12[i * half_size + j] = matrixB[i * new_size + j + half_size];
      B21[i * half_size + j] = matrixB[(i + half_size) * new_size + j];
      B22[i * half_size + j] = matrixB[(i + half_size) * new_size + j + half_size];
    }
  }

  std::vector<std::vector<double>> M(7);
  std::vector<std::vector<double>> tasks = {matrix_add(A11, A22, half_size),
                                            matrix_add(A21, A22, half_size),
                                            A11,
                                            A22,
                                            matrix_add(A11, A12, half_size),
                                            matrix_subtract(A21, A11, half_size),
                                            matrix_subtract(A12, A22, half_size)};

  std::vector<std::vector<double>> tasksB = {
      matrix_add(B11, B22, half_size),      B11, matrix_subtract(B12, B22, half_size),
      matrix_subtract(B21, B11, half_size), B22, matrix_add(B11, B12, half_size),
      matrix_add(B21, B22, half_size)};

  for (int i = 0; i < 7; ++i) {
    M[i] = strassen_recursive(tasks[i], tasksB[i], half_size);
  }

  for (int i = 0; i < 7; ++i) {
    std::vector<double> result;
    M[i] = result;
  }

  std::vector<double> C11 =
      matrix_add(matrix_subtract(matrix_add(M[0], M[3], half_size), M[4], half_size), M[6], half_size);
  std::vector<double> C12 = matrix_add(M[2], M[4], half_size);
  std::vector<double> C21 = matrix_add(M[1], M[3], half_size);
  std::vector<double> C22 =
      matrix_add(matrix_subtract(matrix_add(M[0], M[2], half_size), M[1], half_size), M[5], half_size);

  std::vector<double> result(size * size);
  for (size_t i = 0; i < half_size; ++i) {
    for (size_t j = 0; j < half_size; ++j) {
      result[i * size + j] = C11[i * half_size + j];
      result[i * size + j + half_size] = C12[i * half_size + j];
      result[(i + half_size) * size + j] = C21[i * half_size + j];
      result[(i + half_size) * size + j + half_size] = C22[i * half_size + j];
    }
  }
  return result;
}

std::vector<double> StrassenAlgorithmMPI::strassen_multiply(const std::vector<double>& matrixA,
                                                            const std::vector<double>& matrixB, size_t size) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int rank = world.rank();
  int num_procs = world.size();

  if (size == 1) {
    return {matrixA[0] * matrixB[0]};
  }

  size_t new_size = 1;
  while (new_size < size) {
    new_size *= 2;
  }

  size_t half_size = new_size / 2;

  std::vector<double> A11(half_size * half_size);
  std::vector<double> A12(half_size * half_size);
  std::vector<double> A21(half_size * half_size);
  std::vector<double> A22(half_size * half_size);

  std::vector<double> B11(half_size * half_size);
  std::vector<double> B12(half_size * half_size);
  std::vector<double> B21(half_size * half_size);
  std::vector<double> B22(half_size * half_size);

  for (size_t i = 0; i < half_size; ++i) {
    for (size_t j = 0; j < half_size; ++j) {
      A11[i * half_size + j] = matrixA[i * new_size + j];
      A12[i * half_size + j] = matrixA[i * new_size + j + half_size];
      A21[i * half_size + j] = matrixA[(i + half_size) * new_size + j];
      A22[i * half_size + j] = matrixA[(i + half_size) * new_size + j + half_size];

      B11[i * half_size + j] = matrixB[i * new_size + j];
      B12[i * half_size + j] = matrixB[i * new_size + j + half_size];
      B21[i * half_size + j] = matrixB[(i + half_size) * new_size + j];
      B22[i * half_size + j] = matrixB[(i + half_size) * new_size + j + half_size];
    }
  }

  std::vector<std::vector<double>> M(7);
  if (rank == 0) {
    std::vector<std::vector<double>> tasks = {matrix_add(A11, A22, half_size),
                                              matrix_add(A21, A22, half_size),
                                              A11,
                                              A22,
                                              matrix_add(A11, A12, half_size),
                                              matrix_subtract(A21, A11, half_size),
                                              matrix_subtract(A12, A22, half_size)};

    std::vector<std::vector<double>> tasksB = {
        matrix_add(B11, B22, half_size),      B11, matrix_subtract(B12, B22, half_size),
        matrix_subtract(B21, B11, half_size), B22, matrix_add(B11, B12, half_size),
        matrix_add(B21, B22, half_size)};

    for (int i = 0; i < 7; ++i) {
      if (i % num_procs == 0) {
        M[i] = strassen_recursive(tasks[i], tasksB[i], half_size);
      } else {
        world.send(i % num_procs, i, tasks[i]);
        world.send(i % num_procs, i, tasksB[i]);
      }
    }
  }

  for (int i = 0; i < 7; ++i) {
    if (i % num_procs == rank && i % num_procs != 0) {
      std::vector<double> taskA;
      std::vector<double> taskB;

      world.recv(0, i, taskA);
      world.recv(0, i, taskB);

      M[i] = strassen_recursive(taskA, taskB, half_size);

      world.send(0, i, M[i]);
    }
  }

  if (rank == 0) {
    for (int i = 0; i < 7; ++i) {
      if (i % num_procs != 0) {
        std::vector<double> result;
        world.recv(i % num_procs, i, result);
        M[i] = result;
      }
    }
  }

  if (rank == 0) {
    std::vector<double> C11 =
        matrix_add(matrix_subtract(matrix_add(M[0], M[3], half_size), M[4], half_size), M[6], half_size);
    std::vector<double> C12 = matrix_add(M[2], M[4], half_size);
    std::vector<double> C21 = matrix_add(M[1], M[3], half_size);
    std::vector<double> C22 =
        matrix_add(matrix_subtract(matrix_add(M[0], M[2], half_size), M[1], half_size), M[5], half_size);

    std::vector<double> result(size * size);
    for (size_t i = 0; i < half_size; ++i) {
      for (size_t j = 0; j < half_size; ++j) {
        result[i * size + j] = C11[i * half_size + j];
        result[i * size + j + half_size] = C12[i * half_size + j];
        result[(i + half_size) * size + j] = C21[i * half_size + j];
        result[(i + half_size) * size + j + half_size] = C22[i * half_size + j];
      }
    }
    return result;
  }
  return {};
}

}  // namespace nasedkin_e_strassen_algorithm