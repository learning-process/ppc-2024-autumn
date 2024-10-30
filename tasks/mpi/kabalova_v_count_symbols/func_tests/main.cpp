// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kabalova_v_count_symbols/include/count_symbols_mpi.hpp"

TEST(kabalova_v_count_symbols_mpi, EmptyString) {
  boost::mpi::communicator world;
  std::string global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_count_symbols_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(0, global_out[0]);
  }
}

TEST(kabalova_v_count_symbols_mpi, FourSymbolStringNotLetter) {
  boost::mpi::communicator world;
  std::string global_str = "1234";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kabalova_v_count_symbols_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kabalova_v_count_symbols_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(kabalova_v_count_symbols_mpi, FourSymbolStringLetter) {
  boost::mpi::communicator world;
  std::string global_str = "abcd";

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kabalova_v_count_symbols_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kabalova_v_count_symbols_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}

TEST(kabalova_v_count_symbols_mpi, RandomString) {
  boost::mpi::communicator world;
  std::string global_str = kabalova_v_count_symbols_mpi::getRandomString();
  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kabalova_v_count_symbols_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kabalova_v_count_symbols_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq);
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
