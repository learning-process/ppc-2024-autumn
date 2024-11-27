#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/alputov_i_topology_hypercube/include/ops_mpi.hpp"

namespace alputov_i_topology_hypercube_mpi {

std::vector<int> removeNegativeOnesFromEnd(std::vector<int> vector) {
  auto it = std::find_if_not(vector.rbegin(), vector.rend(), [](int value) { return value == -1; });
  vector.erase(it.base(), vector.end());
  return vector;
}

}  // namespace alputov_i_topology_hypercube_mpi

TEST(alputov_i_topology_hypercube_mpi, DataTransfer_0_to_1) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize >= 2) {
    std::vector<int> inputData{1337, 1};
    std::vector<int> expectedRoute{0, 1};
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}

TEST(alputov_i_topology_hypercube_mpi, DataTransfer_0_to_3) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize >= 4) {
    std::vector<int> inputData{1337, 3};
    std::vector<int> expectedRoute{0, 2, 3};
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}
TEST(alputov_i_topology_hypercube_mpi, DataTransfer_0_to_5) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize >= 8) {
    std::vector<int> inputData{1337, 5};
    std::vector<int> expectedRoute{0, 4, 5};
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}
TEST(alputov_i_topology_hypercube_mpi, DataTransfer_0_to_42) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize > 64) {
    std::vector<int> inputData{1337, 42};
    std::vector<int> expectedRoute{0, 32, 40, 42};
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}

TEST(alputov_i_topology_hypercube_mpi, TargetRankOutOfBounds) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize <= 64) {
    std::vector<int> inputData{1337, 99};
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    if (world.rank() == 0) {
      ASSERT_EQ(TopologyHypercubeMPI.validation(), false);
    }
  }
}

TEST(alputov_i_topology_hypercube_mpi, NegativeTargetRank) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize > 1) {
    std::vector<int> inputData{1337, -5};  // negative rank
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    if (world.rank() == 0) {
      ASSERT_EQ(TopologyHypercubeMPI.validation(), false);
    }
  }
}

TEST(alputov_i_topology_hypercube_mpi, SelfTransfer_Rank0) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if ((communicatorSize & (communicatorSize - 1)) == 0u && communicatorSize >= 1) {
    std::vector<int> inputData{1337, 0};
    std::vector<int> expectedRoute{0};
    std::vector<int> outputData(1, 0);
    std::vector<int> actualRoute;

    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    alputov_i_topology_hypercube_mpi::HypercubeRouterMPI TopologyHypercubeMPI(taskData);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}
