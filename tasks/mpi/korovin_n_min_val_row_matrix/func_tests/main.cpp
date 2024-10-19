// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/korovin_n_min_val_row_matrix/include/ops_mpi.hpp"

TEST(korovin_n_min_val_row_matrix_mpi, find_min_val_in_row_10x10_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 10;
  const int count_columns = 10;

  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(count_rows, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_min(count_rows, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], -25);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, find_min_val_in_row_100x100_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 100;
  const int count_columns = 100;

  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(count_rows, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_min(count_rows, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], -25);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, find_min_val_in_row_100x500_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 100;
  const int count_columns = 500;

  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(count_rows, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_min(count_rows, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], -25);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, find_min_val_in_row_5000x5000_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 5000;
  const int count_columns = 5000;

  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(count_rows, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count = {count_rows, count_columns};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_min(count_rows, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count = {count_rows, count_columns};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], -25);
    }
  }
}