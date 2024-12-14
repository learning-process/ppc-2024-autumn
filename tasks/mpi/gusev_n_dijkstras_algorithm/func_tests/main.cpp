#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "mpi/gusev_n_dijkstras_algorithm/include/ops_mpi.hpp"

TEST(gusev_n_dijkstras_algorithm_mpi, TestDijkstra) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(5);
  graph->add_edge(0, 1, 2.0);
  graph->add_edge(0, 2, 3.0);
  graph->add_edge(1, 2, 1.0);
  graph->add_edge(2, 0, 4.0);
  graph->add_edge(3, 4, 5.0);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));

  std::vector<double> output_data(5);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    std::vector<double> expected_distances = {0.0, 2.0, 3.0, std::numeric_limits<double>::infinity(),
                                              std::numeric_limits<double>::infinity()};

    /*std::cout << "Calculated distances: ";
    for (const auto& dist : output_data) {
      std::cout << dist << " ";
    }
    std::cout << std::endl;*/

    for (size_t i = 0; i < expected_distances.size(); ++i) {
      if (std::isinf(expected_distances[i])) {
        EXPECT_TRUE(std::isinf(output_data[i])) << "Distance to vertex " << i << " should be infinity";
      } else {
        EXPECT_NEAR(output_data[i], expected_distances[i], 1e-6) << "Incorrect distance to vertex " << i;
      }
    }
  }
}

TEST(gusev_n_dijkstras_algorithm_mpi, TestDijkstraDisconnectedGraph) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(5);

  graph->add_edge(0, 1, 2.0);
  graph->add_edge(1, 2, 3.0);
  graph->add_edge(3, 4, 1.0);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));

  std::vector<double> output_data(5);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    std::vector<double> expected_distances = {0.0, 2.0, 5.0, std::numeric_limits<double>::infinity(),
                                              std::numeric_limits<double>::infinity()};

    for (size_t i = 0; i < expected_distances.size(); ++i) {
      if (std::isinf(expected_distances[i])) {
        EXPECT_TRUE(std::isinf(output_data[i])) << "Distance to vertex " << i << " should be infinity";
      } else {
        EXPECT_NEAR(output_data[i], expected_distances[i], 1e-6) << "Incorrect distance to vertex " << i;
      }
    }
  }
}
TEST(gusev_n_dijkstras_algorithm_mpi, TestEmptyGraphValidation) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(0);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));

  std::vector<double> output_data(0);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_FALSE(task.validation()) << "Validation should fail for empty graph";
}

TEST(gusev_n_dijkstras_algorithm_mpi, TestNoInputDataValidation) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<double> output_data(5);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_FALSE(task.validation()) << "Validation should fail with no input data";
}

TEST(gusev_n_dijkstras_algorithm_mpi, TestMismatchedInputSizesValidation) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(5);
  graph->add_edge(0, 1, 2.0);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));

  std::vector<double> output_data(5);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_FALSE(task.validation()) << "Validation should fail with mismatched input sizes";
}

TEST(gusev_n_dijkstras_algorithm_mpi, TestNoOutputDataValidation) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(5);
  graph->add_edge(0, 1, 2.0);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_FALSE(task.validation()) << "Validation should fail with no output data";
}
