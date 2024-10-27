// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_mpi_korobeinikov.hpp"


TEST(max_elements_in_rows_of_matrix_mpi, Test_1_const__matrix) {
  boost::mpi::communicator world;

  // Create data

  int count_rows = 4;  // not const, because reinterpret_cast does not work with const
  std::vector<int> global_matrix{3, 17, 5, -1, 2, -3, 11, 12, 13, -7, 4, 9};
  std::vector<int> mpi_res(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {

    //global_matrix = korobeinikov_a_test_task_mpi::getRandomVector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataPar->inputs_count.emplace_back(1);


    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpi_res.data()));
    taskDataPar->outputs_count.emplace_back(mpi_res.size());
  }

  korobeinikov_a_test_task_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data

    std::vector<int> right_answer(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(right_answer.data()));
    taskDataSeq->outputs_count.emplace_back(right_answer.size());

    // Create Task
    korobeinikov_a_test_task_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(right_answer, mpi_res);
  }
}



TEST(max_elements_in_rows_of_matrix_mpi, Test_2_random_matrix) {
  boost::mpi::communicator world;

  // Create data

  int count_rows = 10;  // not const, because reinterpret_cast does not work with const
  int size_rows = 20;
  std::vector<int> global_matrix{3, 17, 5, -1, 2, -3, 11, 12, 13, -7, 4, 9};
  std::vector<int> mpi_res(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = korobeinikov_a_test_task_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpi_res.data()));
    taskDataPar->outputs_count.emplace_back(mpi_res.size());
  }

  korobeinikov_a_test_task_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data

    std::vector<int> right_answer(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(right_answer.data()));
    taskDataSeq->outputs_count.emplace_back(right_answer.size());

    // Create Task
    korobeinikov_a_test_task_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(right_answer, mpi_res);
  }
}



//TEST(Parallel_Operations_MPI, Test_Diff) {
//  boost::mpi::communicator world;
//  std::vector<int> global_vec;
//  std::vector<int32_t> global_diff(1, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 240;
//    global_vec = nesterov_a_test_task_mpi::getRandomVector(count_size_vector);
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
//    taskDataPar->outputs_count.emplace_back(global_diff.size());
//  }
//
//  nesterov_a_test_task_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<int32_t> reference_diff(1, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_diff.size());
//
//    // Create Task
//    nesterov_a_test_task_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    ASSERT_EQ(reference_diff[0], global_diff[0]);
//  }
//}
//
//TEST(Parallel_Operations_MPI, Test_Diff_2) {
//  boost::mpi::communicator world;
//  std::vector<int> global_vec;
//  std::vector<int32_t> global_diff(1, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 120;
//    global_vec = nesterov_a_test_task_mpi::getRandomVector(count_size_vector);
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
//    taskDataPar->outputs_count.emplace_back(global_diff.size());
//  }
//
//  nesterov_a_test_task_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<int32_t> reference_diff(1, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_diff.size());
//
//    // Create Task
//    nesterov_a_test_task_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    ASSERT_EQ(reference_diff[0], global_diff[0]);
//  }
//}
//
//TEST(Parallel_Operations_MPI, Test_Max) {
//  boost::mpi::communicator world;
//  std::vector<int> global_vec;
//  std::vector<int32_t> global_max(1, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 240;
//    global_vec = nesterov_a_test_task_mpi::getRandomVector(count_size_vector);
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//
//  nesterov_a_test_task_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<int32_t> reference_max(1, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_max.size());
//
//    // Create Task
//    nesterov_a_test_task_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    ASSERT_EQ(reference_max[0], global_max[0]);
//  }
//}
//
//TEST(Parallel_Operations_MPI, Test_Max_2) {
//  boost::mpi::communicator world;
//  std::vector<int> global_vec;
//  std::vector<int32_t> global_max(1, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 120;
//    global_vec = nesterov_a_test_task_mpi::getRandomVector(count_size_vector);
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//
//  nesterov_a_test_task_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<int32_t> reference_max(1, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_max.size());
//
//    // Create Task
//    nesterov_a_test_task_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
//    ASSERT_EQ(testMpiTaskSequential.validation(), true);
//    testMpiTaskSequential.pre_processing();
//    testMpiTaskSequential.run();
//    testMpiTaskSequential.post_processing();
//
//    ASSERT_EQ(reference_max[0], global_max[0]);
//  }
//}

