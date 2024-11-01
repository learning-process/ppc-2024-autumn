// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include "mpi/sedova_o_max_of_vector_elements/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_CanCreate_10) {
  EXPECT_NO_THROW(sedova_o_max_of_vector_elements_mpi::generate_random_matrix(10, 10, 10));
}

TEST(Parallel_Operations_MPI, Test_SquareMatrix_10) {
  const int size = 10;
  const int value = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -(int)value);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(size, size, value);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_SquareMatrix_50) {
  const int size = 50;
  const int value = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -(int)value);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(size, size, spread);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_SquareMatrix_100) {
  const int size = 200;
  const int value = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -(int)value);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(size, size, spread);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_Matrix_10_50) {
  const int rows = 10;
  const int cols = 50;
  const int value = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -(int)value);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_Matrix_100_50) {
  const size_t row = 100;
  const size_t col = 50;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(row, col, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_SquareMatrix_50_WithSeveralMax) {
  const int size = 50;
  const int spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -(int)value);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(size, size, value);
    global_matrix[0][0] = value;
    global_matrix[5][25] = value;
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_Matrix_100_50_WithSeveralMax) {
  const int rows = 100;
  const int cols = 50;
  const int value = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -(int)value);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = sedova_o_max_of_vector_elements_mpi::generate_random_matrix(rows, cols, value);
    global_matrix[25][10] = value;
    global_matrix[5][25] = value;
    for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}