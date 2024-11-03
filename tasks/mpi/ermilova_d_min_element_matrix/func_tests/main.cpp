// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <random>

#include "mpi/ermilova_d_min_element_matrix/include/ops_mpi.hpp"

std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrix(int rows, int cols, int upper_border,
                                                                                 int lower_border) {
  if (rows <= 0 || cols <= 0) throw "Incorrect size";
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(cols, upper_border, lower_border);
  }
  return vec;
}

TEST(ermilova_d_min_element_matrix_mpi, Can_create_vector) {
  const int size_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_mpi, Cant_create_incorrect_size_vector) {
  const int size_test = -10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_mpi, Can_create_matrix) {
  const int rows_test = 10;
  const int cols_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_mpi, Cant_create_incorrect_size_matrix) {
  const int rows_test = -10;
  const int cols_test = 0;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_min_element_matrix_mpi, Matrix_1x1) {
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

  ermilova_d_min_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(rows_test);
    taskDataSeq->inputs_count.emplace_back(cols_test);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    ermilova_d_min_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(ermilova_d_min_element_matrix_mpi, Matrix_10x10) {
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

  ermilova_d_min_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(rows_test);
    taskDataSeq->inputs_count.emplace_back(cols_test);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    ermilova_d_min_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(ermilova_d_min_element_matrix_mpi, Matrix_100x100) {
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

  ermilova_d_min_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(rows_test);
    taskDataSeq->inputs_count.emplace_back(cols_test);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    ermilova_d_min_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(ermilova_d_min_element_matrix_mpi, Matrix_100x50) {
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

  ermilova_d_min_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(rows_test);
    taskDataSeq->inputs_count.emplace_back(cols_test);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    ermilova_d_min_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(ermilova_d_min_element_matrix_mpi, Matrix_50x100) {
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

  ermilova_d_min_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(rows_test);
    taskDataSeq->inputs_count.emplace_back(cols_test);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    ermilova_d_min_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(ermilova_d_min_element_matrix_mpi, Matrix_500x500) {
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

  ermilova_d_min_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(rows_test);
    taskDataSeq->inputs_count.emplace_back(cols_test);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    ermilova_d_min_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}
