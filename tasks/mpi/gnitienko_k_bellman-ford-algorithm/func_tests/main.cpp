#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/gnitienko_k_bellman-ford-algorithm/include/ops_mpi.hpp"

namespace gnitienko_k_generate_func_mpi {

const int MIN_WEIGHT = -5;
const int MAX_WEIGHT = 10;
std::vector<int> generateGraph(const int V) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dis(MIN_WEIGHT, MAX_WEIGHT);

  std::vector<int> graph(V * V, 0);

  for (int i = 0; i < V; ++i) {
    for (int j = i + 1; j < V; ++j) {
      int weight = dis(gen);
      graph[i * V + j] = weight;
    }
  }

  return graph;
}
}  // namespace gnitienko_k_generate_func_mpi

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_simple_graph) {
  boost::mpi::communicator world;
  const int V = 6;
  const int E = 8;

  // Create data
  std::vector<int> graph = {0, 10, 0,  0, 0, 8, 0, 0,  0, 2,  0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0,  -2, 0, 0, 0, 0, -4, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 5, 5, 7, 9, 8};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSEQ(V, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
    ASSERT_EQ(resSEQ, expected_res);
  }
}

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_negative_cycle) {
  boost::mpi::communicator world;
  const int V = 4;
  const int E = 5;

  std::vector<int> graph = {0, 0, -2, 0, 4, 0, -3, 0, 0, 0, 0, 2, 0, -1, 0, 0};

  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  if (world.rank() == 0)
    ASSERT_FALSE(testMpiTaskParallel.run());
  else
    ASSERT_TRUE(testMpiTaskParallel.run());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSEQ(V, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    ASSERT_FALSE(testMpiTaskSequential.run());
  }
}

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_empty_graph) {
  boost::mpi::communicator world;
  const int V = 0;
  const int E = 0;

  std::vector<int> graph = {};

  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0)
    ASSERT_FALSE(testMpiTaskParallel.validation());
  else
    ASSERT_TRUE(testMpiTaskParallel.validation());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSEQ(V, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_2nd_graph) {
  boost::mpi::communicator world;
  const int V = 5;
  const int E = 6;

  std::vector<int> graph = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 9, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0};

  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 10, 5, 11, 14};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> resSEQ(V, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
    ASSERT_EQ(resSEQ, expected_res);
  }
}

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_random_graph) {
  boost::mpi::communicator world;
  const int V = 10;
  int E = 0;

  std::vector<int> graph;

  std::vector<int> resMPI(V, 0);
  std::vector<int> resSEQ;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    graph = gnitienko_k_generate_func_mpi::generateGraph(V);
    for (size_t i = 0; i < graph.size(); i++)
      if (graph[i] != 0) E++;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  bool skip_test = false;
  if (!testMpiTaskParallel.run()) {
    skip_test = true;
  }
  boost::mpi::broadcast(world, skip_test, 0);
  if (skip_test) GTEST_SKIP();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    resSEQ.resize(V);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
  }
}