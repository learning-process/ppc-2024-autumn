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

    for (int i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], INT_MIN);
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

    for (int i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], INT_MIN);
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

    for (int i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], INT_MIN);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, find_min_val_in_row_3000x3000_matrix) {
  boost::mpi::communicator world;
  const int count_rows = 3000;
  const int count_columns = 3000;

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

    for (int i = 0; i < count_rows; i++) {
      ASSERT_EQ(global_min[i], INT_MIN);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, validation_input_empty_100x100_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 100;
    const int cols = 100;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matrix_rnd =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(rows, cols);

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> v_res(rows, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, validation_output_empty_100x100_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 100;
    const int cols = 100;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matrix_rnd =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(rows, cols);

    for (auto& row : matrix_rnd) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> v_res(rows, 0);
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, validation_less_two_cols_100x100_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 100;
    const int cols = 100;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matrix_rnd =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(rows, cols);

    for (auto& row : matrix_rnd) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> v_res(rows, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, validation_find_min_val_in_row_0x10_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 0;
    const int cols = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matrix_rnd =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(rows, cols);

    for (auto& row : matrix_rnd) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> v_res(rows, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, validation_find_min_val_in_row_10x10_cols_0_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 10;
    const int cols = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matrix_rnd =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(rows, cols);

    for (auto& row : matrix_rnd) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(0);

    std::vector<int> v_res(rows, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(korovin_n_min_val_row_matrix_mpi, validation_fails_on_invalid_output_size) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 10;
    const int cols = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    std::vector<std::vector<int>> matrix_rnd =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(rows, cols);

    for (auto& row : matrix_rnd) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> v_res(rows - 1, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}
