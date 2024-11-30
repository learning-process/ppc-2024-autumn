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

namespace golovkin_rowwise_matrix_partitioning {

    void get_random_matrix(double* matr, int size) {
        if (size <= 0) {
            throw std::logic_error("wrong matrix size");
        }
        std::srand(static_cast<unsigned int>(std::time(0)));
        for (int i = 0; i < size; i++) {
            matr[i] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

}  // namespace golovkin_rowwise_matrix_partitioning

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_negative_sizes) {
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

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, invalid_matrix_size) {
  boost::mpi::communicator world;
  if (world.size() < 5 || world.rank() >= 4) {
    int rows_A = 0;
    int cols_A = 10;
    std::unique_ptr<double[]> A(new double[rows_A * cols_A]);
    ASSERT_ANY_THROW(golovkin_rowwise_matrix_partitioning::get_random_matrix(A.get(), rows_A * cols_A));
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, cant_mult_matrix_wrong_sizes) {
  boost::mpi::communicator world;
  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int rows_A = 2;
  int cols_A = 3;
  int rows_B = 7;
  int cols_B = 4;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A * cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B * cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);
    res = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }
  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  if (world.size() < 5 || world.rank() >= 4) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, mult_matrix_and_vector) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int rows_A = 70;
  int cols_A = 50;
  int rows_B = 50;
  int cols_B = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A * cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B * cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    res = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.size() < 5 || world.rank() >= 4) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataSeq->inputs_count.emplace_back(rows_A);
    taskDataSeq->inputs_count.emplace_back(cols_A);
    taskDataSeq->inputs_count.emplace_back(rows_B);
    taskDataSeq->inputs_count.emplace_back(cols_B);

    auto* res_seq = new double[rows_A * cols_B];
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq));
    taskDataSeq->outputs_count.emplace_back(rows_A);
    taskDataSeq->outputs_count.emplace_back(cols_B);

    auto TestTaskSequential =
        std::make_shared<golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask>(taskDataSeq);
    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < rows_A * cols_B; i++) {
      ASSERT_NEAR(res_seq[i], res[i], 1e-6);
    }

    delete[] res_seq;
  }
  if (world.size() < 5 || world.rank() >= 4) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_multiplication_invalid_size) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int rows_A = 3;
  int cols_A = 2;
  int rows_B = 3;
  int cols_B = 3;  // Ошибка, строки A != столбцы B

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A * cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B * cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    res = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  if (world.size() < 5 || world.rank() >= 4) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

TEST(golovkin_rowwise_matrix_partitioning_mpi, matrix_multiplication_with_zeros) {
  boost::mpi::communicator world;

  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int rows_A = 3;
  int cols_A = 3;
  int rows_B = 3;
  int cols_B = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A]{0};  // Инициализация матрицы нулями
    B = new double[rows_B * cols_B];
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B * cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    res = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.size() < 5 || world.rank() >= 4) {
    delete[] A;
    delete[] B;
    delete[] res;
  }
}

