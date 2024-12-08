// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/lavrentyev_a_alternation_count/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_vector_with_one_elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec.push_back(1);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 0);
  }
}

TEST(Parallel_Operations_MPI, Test_two_elems) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec.push_back(-1);
    global_vec.push_back(1);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 1);
  }
}

TEST(Parallel_Operations_MPI, Test_vector_with_zero) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec.push_back(-1);
    global_vec.push_back(0);
    global_vec.push_back(1);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 2);
  }
}

TEST(Parallel_Operations_MPI, Test_vector_with_one_sign_elems) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec.push_back(1);
    global_vec.push_back(1);
    global_vec.push_back(1);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 0);
  }
}

TEST(Parallel_Operations_MPI, Test_big_vector_with_zero) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < 100; i++) {
      global_vec.push_back(-1);
      global_vec.push_back(1);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 199);
  }
}

TEST(Parallel_Operations_MPI, Big_vec_with_zero) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < 100; i++) {
      global_vec.push_back(-1);
      global_vec.push_back(0);
      global_vec.push_back(1);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 299);
  }
}

TEST(Parallel_Operations_MPI, Big_random_vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < 1000; i++) {
      int number = -1000 + rand() % 1000;
      global_vec.push_back(number);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataMpi->inputs_count.emplace_back(global_vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  lavrentyev_a_alternation_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    lavrentyev_a_alternation_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}