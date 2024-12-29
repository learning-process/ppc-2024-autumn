// Golovkin Maksims Task#2
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mpi/golovkin_rowwise_matrix_partitioning/include/ops_mpi.hpp"

using namespace std::chrono_literals;
using namespace golovkin_rowwise_matrix_partitioning;
using ppc::core::TaskData;

namespace golovkin_rowwise_matrix_partitioning {

void get_random_matrix(double *matr, int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::logic_error("wrong matrix size");
  }
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matr[i * cols + j] = static_cast<double>(std::rand()) / RAND_MAX;
    }
  }
}

void sequential_matrix_multiplication(const double *A, const double *B, double *C, int rows_A, int cols_A, int cols_B) {
  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      C[i * cols_B + j] = 0.0;
      for (int k = 0; k < cols_A; ++k) {
        C[i * cols_B + j] += A[i * cols_A + k] * B[k * cols_B + j];
      }
    }
  }
}

}  // namespace golovkin_rowwise_matrix_partitioning

TEST(golovkin_rowwise_matrix_partitioning_mpi, cant_mult_matrix_wrong_sizes) {
  boost::mpi::communicator world;
  double *A = nullptr;
  double *B = nullptr;
  double *result = nullptr;
  int rows_A = 2;
  int cols_A = 3;
  int rows_B = 7;
  int cols_B = 4;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A, cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B, cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);
    result = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if ((world.rank() == 0 && world.size() < 5) || (world.rank() >= 4)) {
    delete[] A;
    delete[] B;
    delete[] result;
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_multiplication_invalid_size) {
  boost::mpi::communicator world;

  double *A = nullptr;
  double *B = nullptr;
  double *result = nullptr;
  int rows_A = 3;
  int cols_A = 2;
  int rows_B = 3;
  int cols_B = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A, cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B, cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if ((world.rank() == 0 && world.size() < 5) || (world.rank() >= 4)) {
    delete[] A;
    delete[] B;
    delete[] result;
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_negative_size) {
  boost::mpi::communicator world;
  int rows_A = -3;
  int cols_A = 3;
  int rows_B = 3;
  int cols_B = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);
  }

  MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_valid_sizes) {
  boost::mpi::communicator world;
  int rows_A = 2;
  int cols_A = 3;
  int rows_B = 3;
  int cols_B = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);
  }

  MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, invalid_matrix_size) {
  boost::mpi::communicator world;
  if (world.size() < 5 || world.rank() >= 4) {
    int rows_A = 0;
    int cols_A = 10;
    std::unique_ptr<double[]> A(new double[rows_A * cols_A]);
    ASSERT_ANY_THROW(golovkin_rowwise_matrix_partitioning::get_random_matrix(A.get(), rows_A, cols_A));
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, initialization_with_empty_inputs) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, invalid_task_with_partial_inputs) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    global_A.resize(100, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(25);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<MPIMatrixMultiplicationTask>(taskDataPar);

  EXPECT_FALSE(taskParallel->validation());
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, invalid_task_with_mismatched_dimensions) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    global_A.resize(25 * 4, 0);

    global_B.resize(3 * 1, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(25);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<MPIMatrixMultiplicationTask>(taskDataPar);

  EXPECT_FALSE(taskParallel->validation());
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_multiplication_correct_result) {
  boost::mpi::communicator world;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> result;
  std::vector<double> expected_result;

  int rows_A = 2;
  int cols_A = 3;
  int rows_B = 3;
  int cols_B = 2;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A.resize(rows_A * cols_A);
    B.resize(rows_B * cols_B);

    golovkin_rowwise_matrix_partitioning::get_random_matrix(A.data(), rows_A, cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B.data(), rows_B, cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result.resize(rows_A * cols_B, 0.0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);

    expected_result.resize(rows_A * cols_B, 0.0);
    golovkin_rowwise_matrix_partitioning::sequential_matrix_multiplication(A.data(), B.data(), expected_result.data(),
                                                                           rows_A, cols_A, cols_B);
  }

  MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  ASSERT_NO_THROW(testMpiTaskParallel.pre_processing());
  ASSERT_NO_THROW(testMpiTaskParallel.run());
  ASSERT_NO_THROW(testMpiTaskParallel.post_processing());

  if ((world.rank() == 0 && world.size() < 5) || (world.rank() >= 4)) {
    if (world.rank() == 0) {
      for (int i = 0; i < rows_A * cols_B; ++i) {
        ASSERT_NEAR(expected_result[i], result[i], 1e-6) << "Mismatch at index " << i;
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_large_sizes) {
  boost::mpi::communicator world;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> result;
  std::vector<double> expected_result;

  int rows_A = 5;
  int cols_A = 6;
  int rows_B = 6;
  int cols_B = 4;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A.resize(rows_A * cols_A);
    B.resize(rows_B * cols_B);

    golovkin_rowwise_matrix_partitioning::get_random_matrix(A.data(), rows_A, cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B.data(), rows_B, cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result.resize(rows_A * cols_B, 0.0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);

    expected_result.resize(rows_A * cols_B, 0.0);
    golovkin_rowwise_matrix_partitioning::sequential_matrix_multiplication(A.data(), B.data(), expected_result.data(),
                                                                           rows_A, cols_A, cols_B);
  }

  MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  ASSERT_NO_THROW(testMpiTaskParallel.pre_processing());
  ASSERT_NO_THROW(testMpiTaskParallel.run());
  ASSERT_NO_THROW(testMpiTaskParallel.post_processing());

  if ((world.rank() == 0 && world.size() < 5) || (world.rank() >= 4)) {
    if (world.rank() == 0) {
      for (int i = 0; i < rows_A * cols_B; ++i) {
        ASSERT_NEAR(expected_result[i], result[i], 1e-6) << "Mismatch at index " << i;
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, memory_leaks_on_failure) {
  boost::mpi::communicator world;

  int rows_A = -3;
  int cols_A = 3;
  int rows_B = 3;
  int cols_B = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);

    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}