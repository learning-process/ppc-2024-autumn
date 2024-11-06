#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {1, 2, 3, 7, 4, 3, 9};
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
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
    ASSERT_EQ(2, global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_450) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 450;
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count);
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
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_1000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 1000;
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count);
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
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_10) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 10;
  std::vector<int32_t> global_violations(1, 0);  // To hold global violations count

  // Create TaskData for Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  // Create the parallel test task
  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task for sequential execution
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 2;
  std::vector<int32_t> global_violations(1, 0);  // To hold global violations count

  // Create TaskData for Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  // Create the parallel test task
  fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task for sequential execution
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}