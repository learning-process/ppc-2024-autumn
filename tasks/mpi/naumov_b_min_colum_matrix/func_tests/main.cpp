// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column) {
  boost::mpi::communicator world;
  const int rows = 12;
  const int cols = 12;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

// Другие тесты
TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column_10x10) {
  boost::mpi::communicator world;
  const int rows = 10;
  const int cols = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column_8x15) {
  boost::mpi::communicator world;
  const int rows = 8;
  const int cols = 15;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Min_Column_5x20) {
  boost::mpi::communicator world;
  const int rows = 5;
  const int cols = 20;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      global_matrix[i] = naumov_b_min_colum_matrix_mpi::getRandomVector(cols);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_minima(cols, std::numeric_limits<int>::max());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minima.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minima.size());

    naumov_b_min_colum_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minima, global_minima);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Invalid_Input_Negative_Rows) {
  boost::mpi::communicator world;
  const int rows = -5;
  const int cols = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(naumov_b_min_colum_matrix_mpi, Test_Invalid_Input_Zero_Cols) {
  boost::mpi::communicator world;
  const int rows = 10;
  const int cols = 0;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(rows, std::vector<int>(cols));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  naumov_b_min_colum_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}
