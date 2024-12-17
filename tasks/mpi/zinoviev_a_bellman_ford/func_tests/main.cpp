// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

TEST(zinoviev_a_bellman_ford, Test_Small_Graph) {
  boost::mpi::communicator world;
  std::vector<int> row_pointers = {0, 1, 2, 3};
  std::vector<int> col_indices = {1, 2, 3};
  std::vector<int> values = {2, 3, 1};
  std::vector<int> distances(4, INT_MAX);
  distances[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_pointers.data()));
  taskDataPar->inputs_count.emplace_back(row_pointers.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indices.data()));
  taskDataPar->inputs_count.emplace_back(col_indices.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values.data()));
  taskDataPar->inputs_count.emplace_back(values.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  taskDataPar->outputs_count.emplace_back(distances.size());

  zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMPITaskParallel(taskDataPar);
  ASSERT_EQ(bellmanFordMPITaskParallel.validation(), true);
  bellmanFordMPITaskParallel.pre_processing();
  bellmanFordMPITaskParallel.run();
  bellmanFordMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_distances = {0, 2, 3, 1};
    ASSERT_EQ(distances, reference_distances);
  }
}