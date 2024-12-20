#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_random_matrices(int n, double min_val = -10.0, double max_val = 10.0) {
  std::vector<std::vector<double>> A(n, std::vector<double>(n));
  std::vector<std::vector<double>> B(n, std::vector<double>(n));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = dist(gen);
      B[i][j] = dist(gen);
    }
  }

  return {A, B};
}

bool check_matrix_multiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, const std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  std::vector<std::vector<double>> expected_C(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        expected_C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (std::abs(expected_C[i][j] - C[i][j]) > 1e-6) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace nasedkin_e_strassen_algorithm_mpi

void run_strassen_test_for_matrix_size(size_t matrix_size) {
  boost::mpi::communicator world;

  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrices(matrix_size);

  std::vector<std::vector<double>> C_parallel(matrix_size, std::vector<double>(matrix_size, 0.0));
  std::vector<std::vector<double>> C_sequential(matrix_size, std::vector<double>(matrix_size, 0.0));

  size_t matrix_size_copy = matrix_size;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size() * A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size() * B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() * C_parallel.size());
  }

  nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmParallel strassen_parallel(taskDataPar);
  ASSERT_TRUE(strassen_parallel.validation());
  strassen_parallel.pre_processing();
  strassen_parallel.run();
  strassen_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size() * A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size() * B.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(C_sequential.size() * C_sequential.size());

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmSequential strassen_sequential(taskDataSeq);
    ASSERT_TRUE(strassen_sequential.validation());
    strassen_sequential.pre_processing();
    strassen_sequential.run();
    strassen_sequential.post_processing();

    ASSERT_TRUE(nasedkin_e_strassen_algorithm_mpi::check_matrix_multiplication(A, B, C_parallel));
    ASSERT_TRUE(nasedkin_e_strassen_algorithm_mpi::check_matrix_multiplication(A, B, C_sequential));
  }
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_2x2) { run_strassen_test_for_matrix_size(2); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_4x4) { run_strassen_test_for_matrix_size(4); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_8x8) { run_strassen_test_for_matrix_size(8); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_16x16) { run_strassen_test_for_matrix_size(16); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_32x32) { run_strassen_test_for_matrix_size(32); }