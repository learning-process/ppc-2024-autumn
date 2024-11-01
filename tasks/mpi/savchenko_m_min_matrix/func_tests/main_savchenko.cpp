// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/savchenko_m_min_matrix/include/ops_mpi_savchenko.hpp"

TEST(savchenko_m_min_matrix_mpi, test_min_10x10) {
  // Create data
  const size_t rows = 10;
  const size_t columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = savchenko_m_min_matrix_mpi::getRandomMatrix(rows, columns, gen_min, gen_max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  savchenko_m_min_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_min.data()));
    taskDataPar->outputs_count.emplace_back(reference_min.size());

    // Create Task
    savchenko_m_min_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(savchenko_m_min_matrix_mpi, test_min_100x10) {
  // Create data
  const size_t rows = 100;
  const size_t columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = savchenko_m_min_matrix_mpi::getRandomMatrix(rows, columns, gen_min, gen_max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  savchenko_m_min_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_min.data()));
    taskDataPar->outputs_count.emplace_back(reference_min.size());

    // Create Task
    savchenko_m_min_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(savchenko_m_min_matrix_mpi, test_min_10x100) {
  // Create data
  const size_t rows = 10;
  const size_t columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = savchenko_m_min_matrix_mpi::getRandomMatrix(rows, columns, gen_min, gen_max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  savchenko_m_min_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_min.data()));
    taskDataPar->outputs_count.emplace_back(reference_min.size());

    // Create Task
    savchenko_m_min_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(savchenko_m_min_matrix_mpi, test_min_100x100) {
  // Create data
  const size_t rows = 100;
  const size_t columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = savchenko_m_min_matrix_mpi::getRandomMatrix(rows, columns, gen_min, gen_max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  savchenko_m_min_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_min.data()));
    taskDataPar->outputs_count.emplace_back(reference_min.size());

    // Create Task
    savchenko_m_min_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}