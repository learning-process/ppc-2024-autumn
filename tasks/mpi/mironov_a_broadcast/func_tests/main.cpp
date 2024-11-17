#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/mironov_a_broadcast/include/ops_mpi.hpp"

TEST(mironov_a_broadcast_mpi, Test_broadcast_1) {
  boost::mpi::communicator world;
  // Create data
  const std::vector<int> golds = {3, 14, 39};
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;
  std::vector<int> reference_res;

  // Global -> custom broadcast, referenced -> boost::mpi
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  // custum broadcust version
  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>({1, 2, 3});
    global_powers = std::vector<int>({1, 2, 3});

    global_res.resize(3, 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // boost::mpi version
  if (world.rank() == 0) {
    // Create TaskData
    reference_res.resize(3, 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataRef->inputs_count.emplace_back(global_input.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataRef->inputs_count.emplace_back(global_powers.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataRef->outputs_count.emplace_back(reference_res.size());
  }
  mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
    ASSERT_EQ(reference_res, golds);
  }
}

TEST(mironov_a_broadcast_mpi, Test_broadcast_2) {
  boost::mpi::communicator world;
  // Create data
  const std::vector<int> golds = {10, 20, 30, 10, 20, 30};
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;
  std::vector<int> reference_res;

  // Global -> custom broadcast, referenced -> boost::mpi
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  // custum broadcust version
  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>({1, 2, 3, 1, 2, 3});
    global_powers = std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // boost::mpi version
  if (world.rank() == 0) {
    // Create TaskData
    reference_res.resize(global_input.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataRef->inputs_count.emplace_back(global_input.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataRef->inputs_count.emplace_back(global_powers.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataRef->outputs_count.emplace_back(reference_res.size());
  }
  mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
    ASSERT_EQ(reference_res, golds);
  }
}

TEST(mironov_a_broadcast_mpi, Test_broadcast_3) {
  boost::mpi::communicator world;
  // Create data
  const std::vector<int> golds = {static_cast<int>(pow(2, 20))};
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;
  std::vector<int> reference_res;

  // Global -> custom broadcast, referenced -> boost::mpi
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  // custum broadcust version
  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>({2});
    global_powers = std::vector<int>({20});

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // boost::mpi version
  if (world.rank() == 0) {
    // Create TaskData
    reference_res.resize(global_input.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataRef->inputs_count.emplace_back(global_input.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataRef->inputs_count.emplace_back(global_powers.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataRef->outputs_count.emplace_back(reference_res.size());
  }
  mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
    ASSERT_EQ(reference_res, golds);
  }
}

TEST(mironov_a_broadcast_mpi, Test_broadcast_4) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;
  std::vector<int> reference_res;

  // Global -> custom broadcast, referenced -> boost::mpi
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  // custum broadcust version
  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>(50);
    global_powers = std::vector<int>(20);
    golds = std::vector<int>(50);

    for (int i = 0; i < 10; ++i) {
      global_powers[2 * i] = i;
      global_powers[2 * i + 1] = i;
    }
    for (int i = 0; i < 50; ++i) {
      global_input[i] = (i + 1) / 10;
      for (auto power : global_powers) {
        golds[i] += static_cast<int>(pow(global_input[i], power));
      }
    }

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // boost::mpi version
  if (world.rank() == 0) {
    // Create TaskData
    reference_res.resize(global_input.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataRef->inputs_count.emplace_back(global_input.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataRef->inputs_count.emplace_back(global_powers.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataRef->outputs_count.emplace_back(reference_res.size());
  }
  mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
    ASSERT_EQ(reference_res, golds);
  }
}
