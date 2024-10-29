#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp"

TEST(sotskov_a_sum_element_matrix, test_constant_matrix) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 1000;
  int cols = 1000;
  std::vector<double> matrix(rows * cols, 5.0);
  double output = 0.0;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataSeq->outputs_count.emplace_back(1);

    sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    double exact = 5.0 * rows * cols;
    EXPECT_NEAR(output, exact, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_random_matrix) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 1000;
  int cols = 1000;
  std::vector<double> matrix(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  double output = 0.0;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataSeq->outputs_count.emplace_back(1);

    sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    double exact = std::accumulate(matrix.begin(), matrix.end(), 0.0);
    EXPECT_NEAR(output, exact, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_empty_matrix) {
  boost::mpi::communicator world;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 0;
  int cols = 0;
  std::vector<double> matrix;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    EXPECT_NEAR(output, 0.0, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_single_element_matrix) {
  boost::mpi::communicator world;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 1;
  int cols = 1;
  std::vector<double> matrix = {7.0};

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    EXPECT_NEAR(output, 7.0, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_zero_matrix) {
  boost::mpi::communicator world;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 10;
  int cols = 10;
  std::vector<double> matrix(rows * cols, 0.0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    EXPECT_NEAR(output, 0.0, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_mixed_values_matrix) {
  boost::mpi::communicator world;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 5;
  int cols = 5;
  std::vector<double> matrix = {1.0,  -1.0, 2.0,  -2.0, 3.0,  -3.0, 4.0,   -4.0, 5.0,   -5.0, 6.0,   -6.0, 7.0,
                                -7.0, 8.0,  -8.0, 9.0,  -9.0, 10.0, -10.0, 11.0, -11.0, 12.0, -12.0, 13.0};

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double exact = std::accumulate(matrix.begin(), matrix.end(), 0.0);
    EXPECT_NEAR(output, exact, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_large_values_matrix) {
  boost::mpi::communicator world;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int rows = 10;
  int cols = 10;
  std::vector<double> matrix(rows * cols, 1e6);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.emplace_back(1);
  }

  sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    double exact = 1e6 * rows * cols;
    EXPECT_NEAR(output, exact, 1e-6);
  }
}

TEST(sotskov_a_sum_element_matrix, test_data_distribution) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int total_rows = 4;
  int total_cols = 3;
  std::vector<double> matrix(total_rows * total_cols);

  if (rank == 0) {
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < total_cols; ++j) {
        matrix[i * total_cols + j] = static_cast<double>(i * total_cols + j + 1);
      }
    }
  }

  boost::mpi::broadcast(world, matrix.data(), matrix.size(), 0);

  int base_elements_per_process = total_rows * total_cols / size;
  int remainder = (total_rows * total_cols) % size;

  int start_idx = rank * base_elements_per_process + std::min(rank, remainder);
  int end_idx = start_idx + base_elements_per_process + (rank < remainder ? 1 : 0);

  for (int i = start_idx; i < end_idx; ++i) {
    double expected_value = i + 1;
    EXPECT_EQ(matrix[i], expected_value) << "Process " << rank << " has incorrect value at index " << i;
  }
}

TEST(sotskov_a_sum_element_matrix, test_data_distribution_single_element_matrix) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int total_rows = 1;
  int total_cols = 1;
  std::vector<double> matrix(total_rows * total_cols, 1.0);

  if (rank == 0) {
    matrix[0] = 42.0;
  }

  boost::mpi::broadcast(world, matrix.data(), matrix.size(), 0);

  int base_elements_per_process = total_rows * total_cols / size;
  int remainder = (total_rows * total_cols) % size;

  int start_idx = rank * base_elements_per_process + std::min(rank, remainder);
  int end_idx = start_idx + base_elements_per_process + (rank < remainder ? 1 : 0);

  if (start_idx < end_idx) {
    EXPECT_EQ(matrix[start_idx], 42.0) << "Process " << rank << " should have value 42.";
  }
}

TEST(sotskov_a_sum_element_matrix, test_data_distribution_2x3_matrix) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int total_rows = 2;
  int total_cols = 3;
  std::vector<double> matrix(total_rows * total_cols);

  if (rank == 0) {
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < total_cols; ++j) {
        matrix[i * total_cols + j] = static_cast<double>(i * total_cols + j + 1);
      }
    }
  }

  boost::mpi::broadcast(world, matrix.data(), matrix.size(), 0);

  int base_elements_per_process = total_rows * total_cols / size;
  int remainder = (total_rows * total_cols) % size;

  int start_idx = rank * base_elements_per_process + std::min(rank, remainder);
  int end_idx = start_idx + base_elements_per_process + (rank < remainder ? 1 : 0);

  for (int i = start_idx; i < end_idx; ++i) {
    double expected_value = i + 1;
    EXPECT_EQ(matrix[i], expected_value) << "Process " << rank << " has incorrect value at index " << i;
  }
}