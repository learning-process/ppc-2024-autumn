#include <gtest/gtest.h>

#include "mpi/durynichev_d_most_different_neighbor_elements/include/ops_mpi.hpp"
namespace durynichev_d_most_different_neighbor_elements_mpi {

std::vector<int> getRandomVector(size_t size) {
  auto device = std::random_device();
  auto generator = std::mt19937(device());
  auto distribution = std::uniform_int_distribution<int>(0, 100'000);
  auto vector = std::vector<int>(size);
  for (auto &val : vector) {
    val = distribution(generator);
  }
  return vector;
}
}  // namespace durynichev_d_most_different_neighbor_elements_mpi

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
    input = durynichev_d_most_different_neighbor_elements_mpi::getRandomVector(200'000);
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
TEST(durynichev_d_most_different_neighbor_elements_mpi, fixed_values_vector) {
  boost::mpi::communicator world;
  std::vector<int> input = {5368, 925, 3210, 500, 4893, 90, 6589, 5367};

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

TEST(durynichev_d_most_different_neighbor_elements_mpi, two_elements) {
  boost::mpi::communicator world;
  std::vector<int> input = {0, 100000};

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
TEST(durynichev_d_most_different_neighbor_elements_mpi, all_elements_equal) {
  boost::mpi::communicator world;
  std::vector<int> input(100, 42);

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

TEST(durynichev_d_most_different_neighbor_elements_mpi, alternating_max_min) {
  boost::mpi::communicator world;
  std::vector<int> input = {100000, 0, 100000, 0, 100000, 0};

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

TEST(durynichev_d_most_different_neighbor_elements_mpi, monotonically_increasing) {
  boost::mpi::communicator world;
  std::vector<int> input = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};

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

TEST(durynichev_d_most_different_neighbor_elements_mpi, repeating_groups) {
  boost::mpi::communicator world;
  std::vector<int> input = {10, 10, 20, 20, 10, 10, 20, 20};

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
