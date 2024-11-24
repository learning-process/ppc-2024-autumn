// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/korotin_e_my_scatter/include/ops_mpi.hpp"

namespace korotin_e_my_scatter_mpi {

std::vector<double> getRandomMatrix(const unsigned rows, const unsigned columns, double scal) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("Can't creaate matrix with 0 rows or columns");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> matrix(rows * columns);
  for (unsigned i = 0; i < rows * columns; i++) {
    matrix[i] = gen() / scal;
  }
  return matrix;
}

}  // namespace korotin_e_my_scatter_mpi

TEST(korotin_e_my_scatter, cant_create_zeroed_matrix) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    ASSERT_ANY_THROW(korotin_e_my_scatter_mpi::getRandomMatrix(0, 10, 100));
    ASSERT_ANY_THROW(korotin_e_my_scatter_mpi::getRandomMatrix(10, 0, 100));
    ASSERT_ANY_THROW(korotin_e_my_scatter_mpi::getRandomMatrix(0, 0, 100));
  }
}

TEST(korotin_e_my_scatter, maxval_is_correct) {
  boost::mpi::communicator world;
  std::vector<double> matrix;
  std::vector<double> max_val(1, 0);
  int root = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == root) {
    const unsigned M = 30;
    const unsigned N = 30;
    matrix = korotin_e_my_scatter_mpi::getRandomMatrix(M, N, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_val.data()));
    taskDataPar->outputs_count.emplace_back(max_val.size());
  }

  korotin_e_my_scatter_mpi::TestMPITaskMyParallel testMpiTaskParallel(taskDataPar, root);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == root) {
    std::vector<double> reference(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    korotin_e_my_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(reference[0], max_val[0]);
  }
}

TEST(korotin_e_my_scatter, matrix_maxval_with_prime_rows_and_columns) {
  boost::mpi::communicator world;
  std::vector<double> matrix;
  std::vector<double> max_val(1, 0);
  int root = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == root) {
    const unsigned M = 29;
    const unsigned N = 31;
    matrix = korotin_e_my_scatter_mpi::getRandomMatrix(M, N, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_val.data()));
    taskDataPar->outputs_count.emplace_back(max_val.size());
  }

  korotin_e_my_scatter_mpi::TestMPITaskMyParallel testMpiTaskParallel(taskDataPar, root);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == root) {
    std::vector<double> reference(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    korotin_e_my_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(reference[0], max_val[0]);
  }
}

TEST(korotin_e_my_scatter, maxval_in_1_1_matrix) {
  boost::mpi::communicator world;
  std::vector<double> matrix;
  std::vector<double> max_val(1, 0);
  int root = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == root) {
    const unsigned M = 1;
    const unsigned N = 1;
    matrix = korotin_e_my_scatter_mpi::getRandomMatrix(M, N, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_val.data()));
    taskDataPar->outputs_count.emplace_back(max_val.size());
  }

  korotin_e_my_scatter_mpi::TestMPITaskMyParallel testMpiTaskParallel(taskDataPar, root);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == root) {
    ASSERT_DOUBLE_EQ(matrix[0], max_val[0]);
  }
}
