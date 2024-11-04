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

  int cols = 0;
  int rows = 0;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = cols * rows;
    matrix = generateRandomVector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Empty_Matrix) {
  boost::mpi::communicator world;

  int cols = 5;
  int rows = 5;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max1) {
  boost::mpi::communicator world;

  int cols = 15;
  int rows = 5;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = cols * rows;
    matrix = generateRandomVector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max2) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 50;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = cols * rows;
    matrix = generateRandomVector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max3) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 100;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = cols * rows;
    matrix = generateRandomVector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max4) {
  boost::mpi::communicator world;

  int cols = 70;
  int rows = 50;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = cols * rows;
    matrix = generateRandomVector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(nasedkin_e_matrix_column_max_value_mpi, Test_Max5) {
  boost::mpi::communicator world;

  int cols = 300;
  int rows = 150;

  std::vector<int> matrix;
  std::vector<int> res_par(cols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = cols * rows;
    matrix = generateRandomVector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(res_par.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    nasedkin_e_matrix_column_max_value_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}