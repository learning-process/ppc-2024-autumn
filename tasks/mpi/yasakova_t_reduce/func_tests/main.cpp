// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/yasakova_t_reduce/include/ops_mpi.hpp"

static std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

static std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int upper_border, int lower_border) {
  if (rows <= 0 || columns <= 0) throw "Incorrect size";
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(columns, upper_border, lower_border);
  }
  return vec;
}

TEST(yasakova_t_reduce, Can_create_valid_vector) {
  const int size_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(yasakova_t_reduce, Cant_create_vector_with_invalid_size) {
  const int size_test = -10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(yasakova_t_reduce, Can_create_valid_matrix) {
  const int rows_test = 10;
  const int cols_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test));
}

TEST(yasakova_t_reduce, Cant_create_matrix_with_invalid_size) {
  const int rows_test = -10;
  const int cols_test = 0;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test));
}

TEST(yasakova_t_reduce, Can_create_valid_1x1_matrix) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);

  const int rows_test = 1;
  const int cols_test = 1;
  const int upper_border_test = 100;
  const int lower_border_test = -100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  yasakova_t_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(yasakova_t_reduce, Can_create_valid_10x10_matrix) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 10;
  const int cols_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  yasakova_t_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(yasakova_t_reduce, Can_create_valid_100x100_matrix) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 100;
  const int cols_test = 100;
  const int upper_border_test = 500;
  const int lower_border_test = -500;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  yasakova_t_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(yasakova_t_reduce, Can_create_valid_100x50_matrix) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 100;
  const int cols_test = 50;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  yasakova_t_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(yasakova_t_reduce, Can_create_valid_50x100_matrix) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 50;
  const int cols_test = 100;
  const int upper_border_test = 500;
  const int lower_border_test = -500;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  yasakova_t_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(yasakova_t_reduce, Can_create_valid_500x500_matrix) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  const int rows_test = 500;
  const int cols_test = 500;
  const int upper_border_test = 500;
  const int lower_border_test = -500;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows_test);
    taskDataPar->inputs_count.emplace_back(cols_test);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  yasakova_t_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int reference_min = INT_MAX;
    for (auto& row : global_matrix) {
      int row_min = *std::min_element(row.begin(), row.end());
      reference_min = std::min(reference_min, row_min);
    }

    ASSERT_EQ(reference_min, global_min[0]);
  }
}