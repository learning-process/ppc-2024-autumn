#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/solovev_a_star_topology/include/ops_mpi.hpp"

TEST(solovev_a_star_topology_mpi, test_empty_input) {
  boost::mpi::communicator world;
  if (world.size() > 1) {
    std::vector<int> input = {};
    std::vector<int> output(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());

      solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
      ASSERT_EQ(testMpiTaskParallel.validation(), false);
    }
  }
}

TEST(solovev_a_star_topology_mpi, Test_Transfer_1) {
  boost::mpi::communicator world;
  if (world.size() > 1) {
    std::vector<int> input = {1};
    std::vector<int> output(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
    }
    solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(output[i], input[i]);
      }
    }
  }
}

TEST(solovev_a_star_topology_mpi, Test_Transfer_3) {
  boost::mpi::communicator world;
  if (world.size() > 1) {
    std::vector<int> input{1, 1, 1}; 
    std::vector<int> output(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
    }
    solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(output[i], input[i]);
      }
    }
  }
}

TEST(solovev_a_star_topology_mpi, Test_Transfer_10) {
  boost::mpi::communicator world;
  if (world.size() > 1) {
    std::vector<int> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> output(10, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
    }
    solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(output[i], input[i]);
      }
    }
  }
}
