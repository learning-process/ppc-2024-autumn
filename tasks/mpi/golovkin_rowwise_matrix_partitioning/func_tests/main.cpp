// Golovkin Maksim
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

TEST(golovkin_rowwise_matrix_partitioning, test_small_matrices) {
  boost::mpi::communicator world;

  const int rows_A = 2, cols_A = 3, rows_B = 3, cols_B = 2;
  double A[rows_A][cols_A] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double B[rows_B][cols_B] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  double computed_result[rows_A][cols_B] = {0};
  double expected_result[rows_A][cols_B] = {{58.0, 64.0}, {139.0, 154.0}};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(rows_A);
    taskData->inputs_count.emplace_back(cols_A);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(rows_B);
    taskData->inputs_count.emplace_back(cols_B);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(rows_A);
    taskData->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < rows_A; ++i) {
      for (int j = 0; j < cols_B; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_identity_matrices) {
  boost::mpi::communicator world;

  const int size = 3;
  double A[size][size] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  double B[size][size] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  double computed_result[size][size] = {0};
  double expected_result[size][size] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(size);
    taskData->inputs_count.emplace_back(size);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(size);
    taskData->inputs_count.emplace_back(size);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_random_matrices) {
  boost::mpi::communicator world;

  const int rows_A = 4, cols_A = 5, rows_B = 5, cols_B = 3;
  double A[rows_A][cols_A];
  double B[rows_B][cols_B];
  double computed_result[rows_A][cols_B] = {0};
  double expected_result[rows_A][cols_B] = {0};

  srand(world.rank());
  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_A; ++j) {
      A[i][j] = rand() % 10;
    }
  }
  for (int i = 0; i < rows_B; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      B[i][j] = rand() % 10;
    }
  }

  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      for (int k = 0; k < cols_A; ++k) {
        expected_result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(rows_A);
    taskData->inputs_count.emplace_back(cols_A);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(rows_B);
    taskData->inputs_count.emplace_back(cols_B);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(rows_A);
    taskData->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < rows_A; ++i) {
      for (int j = 0; j < cols_B; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_multiplication_with_zero_matrix) {
  boost::mpi::communicator world;

  const int rows_A = 3, cols_A = 3, rows_B = 3, cols_B = 3;
  double A[rows_A][cols_A] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  double B[rows_B][cols_B] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  double computed_result[rows_A][cols_B] = {0};
  double expected_result[rows_A][cols_B] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(rows_A);
    taskData->inputs_count.emplace_back(cols_A);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(rows_B);
    taskData->inputs_count.emplace_back(cols_B);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(rows_A);
    taskData->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < rows_A; ++i) {
      for (int j = 0; j < cols_B; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_non_square_matrices) {
  boost::mpi::communicator world;

  const int rows_A = 2, cols_A = 3, rows_B = 3, cols_B = 4;
  double A[rows_A][cols_A] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double B[rows_B][cols_B] = {{7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0}, {15.0, 16.0, 17.0, 18.0}};
  double computed_result[rows_A][cols_B] = {0};
  double expected_result[rows_A][cols_B] = {{74.0, 80.0, 86.0, 92.0}, {173.0, 188.0, 203.0, 218.0}};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(rows_A);
    taskData->inputs_count.emplace_back(cols_A);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(rows_B);
    taskData->inputs_count.emplace_back(cols_B);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(rows_A);
    taskData->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < rows_A; ++i) {
      for (int j = 0; j < cols_B; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_large_matrices) {
  boost::mpi::communicator world;

  const int size = 100;
  double A[size][size];
  double B[size][size];
  double computed_result[size][size] = {0};
  double expected_result[size][size] = {0};

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      A[i][j] = (i + 1) * (j + 1);
      B[i][j] = (i + 1) + (j + 1);
    }
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        expected_result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(size);
    taskData->inputs_count.emplace_back(size);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(size);
    taskData->inputs_count.emplace_back(size);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_negative_elements) {
  boost::mpi::communicator world;

  const int size = 2;
  double A[size][size] = {{-1, -2}, {-3, -4}};
  double B[size][size] = {{-5, -6}, {-7, -8}};
  double computed_result[size][size] = {0};
  double expected_result[size][size] = {{19, 22}, {43, 50}};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskData->inputs_count.emplace_back(size);
    taskData->inputs_count.emplace_back(size);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));
    taskData->inputs_count.emplace_back(size);
    taskData->inputs_count.emplace_back(size);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result));
    taskData->outputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask parallel_task(taskData);
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        ASSERT_NEAR(computed_result[i][j], expected_result[i][j], 1e-3);
      }
    }
  }
}
