// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/koshkin_n_sum_values_by_columns_matrix/include/ops_mpi.hpp"

std::vector<int> koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

TEST(koshkin_n_sum_values_by_columns_matrix_MPI, Test_SquareMatrixSmall) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 10;

  std::vector<int> matrix = koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> res_out_paral(columns, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }

  koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_out_seq(columns, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());

    // Create Task
    koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(koshkin_n_sum_values_by_columns_matrix_MPI, Test_EmptyMatrix) {
  boost::mpi::communicator world;

  const int rows = 0;
  const int columns = 0;

  std::vector<int> matrix = {};
  std::vector<int> res_out_paral(columns, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }

  koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_out_seq(columns, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());

    // Create Task
    koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    // Since I require non-empty data, it is logical that it simply will not pass
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(koshkin_n_sum_values_by_columns_matrix_MPI, Test_SquareMatrixMedium) {
  boost::mpi::communicator world;

  const int rows = 100;
  const int columns = 100;

  std::vector<int> matrix = koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> res_out_paral(columns, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }

  koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_out_seq(columns, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());

    // Create Task
    koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(koshkin_n_sum_values_by_columns_matrix_MPI, Test_SquareMatrixLarge) {
  boost::mpi::communicator world;

  const int rows = 1000;
  const int columns = 1000;

  std::vector<int> matrix = koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> res_out_paral(columns, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }

  koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_out_seq(columns, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());

    // Create Task
    koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(koshkin_n_sum_values_by_columns_matrix_MPI, Test_MatrixSmall15x10) {
  boost::mpi::communicator world;

  const int rows = 15;
  const int columns = 10;

  std::vector<int> matrix = koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> res_out_paral(columns, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }

  koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_out_seq(columns, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());

    // Create Task
    koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(koshkin_n_sum_values_by_columns_matrix_MPI, Test_Matrix500x1000) {
  boost::mpi::communicator world;

  const int rows = 500;
  const int columns = 1000;

  std::vector<int> matrix = koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(columns * rows);
  std::vector<int> res_out_paral(columns, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }

  koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_out_seq(columns, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());

    // Create Task
    koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}