// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/rams_s_char_frequency/include/ops_mpi.hpp"

TEST(rams_s_char_frequency_mpi, several_occurrences_of_target) {
  boost::mpi::communicator world;
  std::string global_in = "abcdabcda";
  std::vector<int> global_in_target(1, 'a');

  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  rams_s_char_frequency_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataSeq->inputs_count.emplace_back(global_in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataSeq->inputs_count.emplace_back(global_in_target.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    rams_s_char_frequency_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(rams_s_char_frequency_mpi, no_occurrences_of_target) {
  boost::mpi::communicator world;
  std::string global_in = "bcdbcd";
  std::vector<int> global_in_target(1, 'a');

  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  rams_s_char_frequency_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataSeq->inputs_count.emplace_back(global_in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataSeq->inputs_count.emplace_back(global_in_target.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    rams_s_char_frequency_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(rams_s_char_frequency_mpi, empty_input_string) {
  boost::mpi::communicator world;
  std::string global_in;
  std::vector<int> global_in_target(1, 'a');

  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  rams_s_char_frequency_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataSeq->inputs_count.emplace_back(global_in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataSeq->inputs_count.emplace_back(global_in_target.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    rams_s_char_frequency_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(rams_s_char_frequency_mpi, large_input_string) {
  boost::mpi::communicator world;
  std::string common_string = "abc";
  std::string global_in;
  for (int i = 0; i < 9999; i++) {
    global_in += common_string;
  }
  std::vector<int> global_in_target(1, 'a');

  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  rams_s_char_frequency_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataSeq->inputs_count.emplace_back(global_in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataSeq->inputs_count.emplace_back(global_in_target.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    rams_s_char_frequency_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(rams_s_char_frequency_mpi, random_input_string) {
  boost::mpi::communicator world;
  std::string chars = "1234567890abcdefghijklmnopqrstuvwxyz!@#$%^&*()";
  std::string global_in;
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < 9999; i++) {
    global_in += chars[gen() % chars.length()];
  }
  std::vector<int> global_in_target(1, 'a');

  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  rams_s_char_frequency_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataSeq->inputs_count.emplace_back(global_in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataSeq->inputs_count.emplace_back(global_in_target.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    rams_s_char_frequency_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
