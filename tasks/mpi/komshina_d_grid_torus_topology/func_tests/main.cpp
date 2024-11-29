
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_topology_mpi, TestLargeNumberOfProcesses) {
  boost::mpi::communicator world;
  if (world.size() < 16) return;

  std::vector<int> input(16);
  std::iota(input.begin(), input.end(), 9);
  std::vector<int> output(16, 0);
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

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel gridTorusTask(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(gridTorusTask.validation(), false);
  } else {
    ASSERT_EQ(gridTorusTask.validation(), true);
    gridTorusTask.pre_processing();
    gridTorusTask.run();
    gridTorusTask.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(output, input);
      ASSERT_EQ(order, real_order);
    }
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestMinimumProcesses) {
  boost::mpi::communicator world;

  if (world.size() != 2) return;

  std::vector<int> input = {1, 2};
  std::vector<int> output(2, 0);
  std::vector<int> order(world.size() + 1, -1);
  std::vector<int> real_order(world.size() + 1);
  std::iota(real_order.begin(), real_order.end(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel gridTorusTask(taskDataPar);
  ASSERT_EQ(gridTorusTask.validation(), true);
  gridTorusTask.pre_processing();
  gridTorusTask.run();
  gridTorusTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output, input);
    ASSERT_EQ(order, real_order);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestScalability) {
  boost::mpi::communicator world;

  if (world.size() < 32) return;

  std::vector<int> input(world.size());
  std::iota(input.begin(), input.end(), 1);  // ������� ������ �� 1 �� size
  std::vector<int> output(world.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel gridTorusTask(taskDataPar);
  ASSERT_EQ(gridTorusTask.validation(), true);
  gridTorusTask.pre_processing();
  gridTorusTask.run();
  gridTorusTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output, input);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestValidationFailsOnMismatchedSizes) {
  boost::mpi::communicator world;

  std::vector<int> input(4, 1);
  std::vector<int> output(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel gridTorusTask(taskDataPar);
  ASSERT_EQ(gridTorusTask.validation(), false);
}