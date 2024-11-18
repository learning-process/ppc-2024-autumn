#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "mpi/solovyev_d_topology_hypercube/include/header.hpp"

namespace solovyev_d_topology_hypercube_mpi {
std::vector<int> removeTrailing(std::vector<int> vec) {
  vec.erase(std::remove(vec.begin(), vec.end(), -1), vec.end());
  return vec;
}
}  // namespace solovyev_d_topology_hypercube_mpi

TEST(solovyev_d_topology_hypercube_mpi, Test_Transfer_0) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) >= 1)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 1};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0, 1};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()) + 1, -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI TopologyHypercubeMPI(taskDataPar);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      outputPath = solovyev_d_topology_hypercube_mpi::removeTrailing(outputPath);
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(solovyev_d_topology_hypercube_mpi, Test_Transfer_1) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) >= 2)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 3};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0, 2, 3};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()) + 1, -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI TopologyHypercubeMPI(taskDataPar);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      outputPath = solovyev_d_topology_hypercube_mpi::removeTrailing(outputPath);
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(solovyev_d_topology_hypercube_mpi, Test_Transfer_2) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) >= 6)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 54};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0, 32, 48, 52, 54};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()) + 1, -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI TopologyHypercubeMPI(taskDataPar);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      outputPath = solovyev_d_topology_hypercube_mpi::removeTrailing(outputPath);
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(solovyev_d_topology_hypercube_mpi, Test_Transfer_Self) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) >= 0)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 0};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()) + 1, -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI TopologyHypercubeMPI(taskDataPar);
    ASSERT_EQ(TopologyHypercubeMPI.validation(), true);
    TopologyHypercubeMPI.pre_processing();
    TopologyHypercubeMPI.run();
    TopologyHypercubeMPI.post_processing();
    if (world.rank() == 0) {
      outputPath = solovyev_d_topology_hypercube_mpi::removeTrailing(outputPath);
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

// test that validation return false if destination bigger than number of processes
TEST(solovyev_d_topology_hypercube_mpi, Test_Transfer_Not_Exist) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) <= 4)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 32};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0, 4};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()), -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI TopologyHypercubeMPI(taskDataPar);
    if (world.rank() == 0) {
      ASSERT_EQ(TopologyHypercubeMPI.validation(), false);
    }
  }
}

int main(int argc, char* argv[]) {
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}