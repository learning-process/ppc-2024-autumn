#include <gtest/gtest.h>

#include "mpi/durynichev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

TEST(durynichev_d_most_different_neighbor_elements_mpi, default_vector) {
  boost::mpi::communicator world;
  std::vector<int> input;

  std::vector<int> outputPar{0, 0};
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = durynichev_d_most_different_neighbor_elements_mpi::getRandomVector(20'000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPar.data()));
    taskDataPar->outputs_count.emplace_back(outputPar.size());
  }

  durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> outputSeq{0, 0};
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outputSeq.size());

    durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(outputSeq, outputPar);
  }
}

TEST(durynichev_d_most_different_neighbor_elements_mpi, huge_vector) {
  boost::mpi::communicator world;
  std::vector<int> input;

  std::vector<int> outputPar{0, 0};
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = durynichev_d_most_different_neighbor_elements_mpi::getRandomVector(100'000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPar.data()));
    taskDataPar->outputs_count.emplace_back(outputPar.size());
  }

  durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> outputSeq{0, 0};
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outputSeq.size());

    durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(outputSeq, outputPar);
  }
}

TEST(durynichev_d_most_different_neighbor_elements_mpi, zero_elements) {
  boost::mpi::communicator world;
  std::vector<int> input(10'000, 0);

  std::vector<int> outputPar{0, 0};
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPar.data()));
    taskDataPar->outputs_count.emplace_back(outputPar.size());
  }

  durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> outputSeq{0, 0};
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outputSeq.size());

    durynichev_d_most_different_neighbor_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(outputSeq, outputPar);
  }
}
