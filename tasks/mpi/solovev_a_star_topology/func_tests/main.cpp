#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/solovev_a_star_topology/include/ops_mpi.hpp"

TEST(solovev_a_star_topology_mpi, Test_order) {
  boost::mpi::communicator world;
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(1);
  std::vector<int> output(1, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::vector<int> real_order(world.size() + 1);
  for (int n = 0; n < world.size() + 1; n++) {
    real_order[n] = n;
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }
  solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(output, input);
      ASSERT_EQ(order, real_order);
    }
  }
}

TEST(solovev_a_star_topology_mpi, Test_empty_input) {
  boost::mpi::communicator world;
  std::vector<int> input = {};
  std::vector<int> output(1, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
    solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(solovev_a_star_topology_mpi, Test_dif_size_input_output) {
  boost::mpi::communicator world;
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(3);
  std::vector<int> output(5, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
    solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(solovev_a_star_topology_mpi, Test_Transfer_1) {
  boost::mpi::communicator world;
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(1);
  std::vector<int> output(1, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }
  solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(output, input);
    }
  }
}

TEST(solovev_a_star_topology_mpi, Test_Transfer_3) {
  boost::mpi::communicator world;
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(3);
  std::vector<int> output(3, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }
  solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(output, input);
    }
  }
}

TEST(solovev_a_star_topology_mpi, Test_Transfer_10) {
  boost::mpi::communicator world;
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(10);
  std::vector<int> output(10, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }
  solovev_a_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(output, input);
    }
  }
}
