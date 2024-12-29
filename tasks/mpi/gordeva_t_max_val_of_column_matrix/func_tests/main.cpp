#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <vector>

#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

TEST(gordeva_t_max_val_of_column_matrix_mpi, IsEmptyInput) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, IsEmptyOutput) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(5);
    taskDataPar->inputs_count.push_back(5);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[25]));
    ASSERT_FALSE(testMpiTaskParallel.validation());

    delete[] reinterpret_cast<int*>(taskDataPar->inputs[0]);
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Max_val_of_500_columns_with_random) {
  boost::mpi::communicator world;

  const int rows = 500;
  const int cols = 500;
  std::vector<std::vector<int32_t>> global_matr;
  std::vector<int32_t> global_max(cols, INT_MIN);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(global_max, max_example);
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Max_val_of_500_1000_columns_with_random) {
  boost::mpi::communicator world;

  const int rows = 500;
  const int cols = 1000;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(cols, INT_MIN);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < cols; i++) {
      ASSERT_EQ(global_max[i], max_example[i]);
    }
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Max_val_of_1000_3000_columns_with_random) {
  boost::mpi::communicator world;

  const int rows = 1000;
  const int cols = 3000;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(cols, INT_MIN);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < cols; i++) {
      ASSERT_EQ(global_max[i], max_example[i]);
    }
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Incorrect_val_size_of_input) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(2);
    taskDataPar->inputs_count.push_back(3);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[6]));
    taskDataPar->outputs_count.push_back(2);
    ASSERT_FALSE(testMpiTaskParallel.validation());

    delete[] reinterpret_cast<int*>(taskDataPar->inputs[0]);
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Incorrect_val_of_output) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(10);
    taskDataPar->inputs_count.push_back(15);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[150]));
    taskDataPar->outputs_count.push_back(2);
    ASSERT_FALSE(testMpiTaskParallel.validation());

    delete[] reinterpret_cast<int*>(taskDataPar->inputs[0]);
  }
}