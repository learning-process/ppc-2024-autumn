#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iomanip>
#include <random>
#include <vector>

#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

std::vector<int> generateRandomVector(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    int value = gen() % 200 - 100;
    if (value >= 0) {
      vec[i] = value;
    }
  }
  return vec;
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Zero_Columns) {
  boost::mpi::communicator world;

  int numCols = 0;
  int numRows = 0;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int totalSize = numCols * numRows;
    matrix = generateRandomVector(totalSize);

    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Empty_Matrix) {
  boost::mpi::communicator world;

  int numCols = 5;
  int numRows = 5;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max1) {
  boost::mpi::communicator world;

  int numCols = 15;
  int numRows = 5;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int totalSize = numCols * numRows;
    matrix = generateRandomVector(totalSize);

    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resultSequential(numCols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSequential->inputs_count.emplace_back(matrix.size());
    taskDataSequential->inputs_count.emplace_back(numCols);
    taskDataSequential->inputs_count.emplace_back(numRows);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSequential.data()));
    taskDataSequential->outputs_count.emplace_back(resultSequential.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSequential);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resultSequential, resultParallel);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max2) {
  boost::mpi::communicator world;

  int numCols = 50;
  int numRows = 50;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int totalSize = numCols * numRows;
    matrix = generateRandomVector(totalSize);

    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resultSequential(numCols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSequential->inputs_count.emplace_back(matrix.size());
    taskDataSequential->inputs_count.emplace_back(numCols);
    taskDataSequential->inputs_count.emplace_back(numRows);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSequential.data()));
    taskDataSequential->outputs_count.emplace_back(resultSequential.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSequential);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resultSequential, resultParallel);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max3) {
  boost::mpi::communicator world;

  int numCols = 50;
  int numRows = 100;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int totalSize = numCols * numRows;
    matrix = generateRandomVector(totalSize);

    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resultSequential(numCols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSequential->inputs_count.emplace_back(matrix.size());
    taskDataSequential->inputs_count.emplace_back(numCols);
    taskDataSequential->inputs_count.emplace_back(numRows);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSequential.data()));
    taskDataSequential->outputs_count.emplace_back(resultSequential.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSequential);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resultSequential, resultParallel);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max4) {
  boost::mpi::communicator world;

  int numCols = 70;
  int numRows = 50;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int totalSize = numCols * numRows;
    matrix = generateRandomVector(totalSize);

    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resultSequential(numCols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSequential->inputs_count.emplace_back(matrix.size());
    taskDataSequential->inputs_count.emplace_back(numCols);
    taskDataSequential->inputs_count.emplace_back(numRows);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSequential.data()));
    taskDataSequential->outputs_count.emplace_back(resultSequential.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSequential);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resultSequential, resultParallel);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max5) {
  boost::mpi::communicator world;

  int numCols = 300;
  int numRows = 150;

  std::vector<int> matrix;
  std::vector<int> resultParallel(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int totalSize = numCols * numRows;
    matrix = generateRandomVector(totalSize);

    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resultSequential(numCols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSequential->inputs_count.emplace_back(matrix.size());
    taskDataSequential->inputs_count.emplace_back(numCols);
    taskDataSequential->inputs_count.emplace_back(numRows);
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSequential.data()));
    taskDataSequential->outputs_count.emplace_back(resultSequential.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSequential);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resultSequential, resultParallel);
  }
}