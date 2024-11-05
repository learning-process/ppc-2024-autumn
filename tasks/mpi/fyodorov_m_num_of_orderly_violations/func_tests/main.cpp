#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_Count_ViolationsDDDDDDDDDDDDDADDAADAD) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);  // To hold global violations count

  // Create TaskData for Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;  // You can change the size as needed
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  // Create the parallel test task
  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create reference data for sequential computation
    std::vector<int32_t> reference_violations(1, 0);

    // Create TaskData for Sequential
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create the sequential test task
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    // Now compare parallel and sequential results
    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}
/*
TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 120;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data for sequential test
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task for sequential execution
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "count");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 240;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data for sequential test
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task for sequential execution
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "count");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}
*/
/*
TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Violations_1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_violations(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}


TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Violations_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 240;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_violations(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Violations_3) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 360;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_violations(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }

}
*/