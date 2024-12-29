#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"

std::vector<int> RandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-100, 100);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

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

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {1, 2, 3, 4, 5, 6, 7};
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
    ASSERT_EQ(0, global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_450) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 450;
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = RandomVector(count);
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
    global_vec = RandomVector(count);
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

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_3000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 3000;
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = RandomVector(count);
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
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = RandomVector(count);
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

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Random_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 2;
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = RandomVector(count);
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

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Violations_Reverse) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
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
    ASSERT_EQ(9, global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, Test_Count_Violations_Same_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {5, 5, 5, 5, 5};
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
    ASSERT_EQ(0, global_violations[0]);
  }
}