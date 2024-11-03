// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/gnitienko_k_sum_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(gnitienko_k_sum_values_by_rows_MPI, test_empty_matrix) {
  boost::mpi::communicator world;

  int rows = 0;
  int cols = 0;

  std::vector<int> global_vec;
  std::vector<int> resMPI;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_of_a_given_matrix) {
  boost::mpi::communicator world;

  int cols = 3;
  int rows = 4;

  std::vector<int> global_vec = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<int> resMPI(rows, 0);
  std::vector<int> expected_sums = {3, 6, 9, 12};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_large_matrix) {
  boost::mpi::communicator world;

  int cols = 1000;
  int rows = 2000;

  std::vector<int> global_vec(rows * cols, 0);
  std::vector<int> resMPI(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_negative_values) {
  boost::mpi::communicator world;

  int cols = 100;
  int rows = 100;

  std::vector<int> global_vec(rows * cols, -1);
  std::vector<int> resMPI(rows, 0);
  std::vector<int> expect(rows, -100);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_output_element) {
  boost::mpi::communicator world;

  int cols = 3;
  int rows = 4;

  std::vector<int> global_vec = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<int> resMPI(rows, 0);
  int expected_out_3 = 12;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI[3], expected_out_3);
    ASSERT_EQ(resSeq[3], expected_out_3);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_random_matrix) {
  boost::mpi::communicator world;

  int cols = 3;
  int rows = 4;

  std::vector<int> global_vec = gnitienko_k_sum_row_mpi::getRandomVector(rows * cols);
  std::vector<int> resMPI(rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, resSeq);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_of_empty_matrix_2) {
  boost::mpi::communicator world;

  int cols = 0;
  int rows = 4;

  std::vector<int> global_vec;
  std::vector<int> resMPI(rows, 0);
  std::vector<int> expected_sums = {0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, expected_sums);
    ASSERT_EQ(resSeq, expected_sums);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_small_matrix) {
  boost::mpi::communicator world;

  int cols = 1;
  int rows = 1;

  std::vector<int> global_vec = {15};
  std::vector<int> resMPI(rows, 0);
  std::vector<int> expect = {15};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, expect);
    ASSERT_EQ(resSeq, expect);
  }
}

TEST(gnitienko_k_sum_values_by_rows_MPI, test_two_columns_one_row) {
  boost::mpi::communicator world;

  int cols = 2;
  int rows = 1;

  std::vector<int> global_vec = {12, 10};
  std::vector<int> resMPI(rows, 0);
  std::vector<int> expect = {22};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(rows);
  }

  gnitienko_k_sum_row_mpi::SumByRowMPIParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSeq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);

    // Create Task
    gnitienko_k_sum_row_mpi::SumByRowMPISeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI, expect);
    ASSERT_EQ(resSeq, expect);
  }
}