#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/gnitienko_k_bellman-ford-algorithm/include/ops_mpi.hpp"

namespace gnitienko_k_generate_func_mpi {

const int MIN_WEIGHT = -5;
const int MAX_WEIGHT = 10;
enum GraphType : std::uint8_t { RANDOM, CYCLIC, MULTIGRAPH };

std::vector<int> generateGraph(const int V, GraphType type, int edges = 0) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dis(MIN_WEIGHT, MAX_WEIGHT);

  std::vector<int> graph(V * V, 0);

  if (type == CYCLIC) {
    edges = V * (V - 1);
  }

  if (type == RANDOM || type == MULTIGRAPH) {
    edges = std::min(edges, V * (V - 1) / 4);
  }

  switch (type) {
    case RANDOM: {
      for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
          int weight = dis(gen);
          graph[i * V + j] = weight;
        }
      }
      break;
    }

    case MULTIGRAPH: {
      for (int e = 0; e < edges; ++e) {
        int u = rand() % V;
        int v = rand() % V;
        int weight = dis(gen);
        graph[u * V + v] += weight;
      }
      break;
    }

    case CYCLIC: {
      for (int i = 0; i < V; ++i) {
        int weight = dis(gen);
        graph[i * V + ((i + 1) % V)] = weight;
      }
      break;
    }
  }
  return graph;
}
}  // namespace gnitienko_k_generate_func_mpi

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_simple_graph) {
  boost::mpi::communicator world;
  const int V = 6;

  // Create data
  std::vector<int> graph = {0, 10, 0,  0, 0, 8, 0, 0,  0, 2,  0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0,  -2, 0, 0, 0, 0, -4, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 5, 5, 7, 9, 8};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

  std::vector<int> graph = {0, 0, -2, 0, 4, 0, -3, 0, 0, 0, 0, 2, 0, -1, 0, 0};

  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

  std::vector<int> graph = {};

  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

  std::vector<int> graph = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 9, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0};

  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 10, 5, 11, 14};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

  std::vector<int> graph;

  std::vector<int> resMPI(V, 0);
  std::vector<int> resSEQ;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    graph = gnitienko_k_generate_func_mpi::generateGraph(V, gnitienko_k_generate_func_mpi::RANDOM, 15);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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
    resSEQ.resize(V);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_loop_graph) {
  boost::mpi::communicator world;
  const int V = 2;

  std::vector<int> graph = {1, 1, 0, 2};

  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_random_graph_20) {
  boost::mpi::communicator world;
  const int V = 20;

  std::vector<int> graph;

  std::vector<int> resMPI(V, 0);
  std::vector<int> resSEQ;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    graph = gnitienko_k_generate_func_mpi::generateGraph(V, gnitienko_k_generate_func_mpi::RANDOM, 30);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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
    resSEQ.resize(V);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_random_graph_40) {
  boost::mpi::communicator world;
  const int V = 40;

  std::vector<int> graph;

  std::vector<int> resMPI(V, 0);
  std::vector<int> resSEQ;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    graph = gnitienko_k_generate_func_mpi::generateGraph(V, gnitienko_k_generate_func_mpi::RANDOM, 50);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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
    resSEQ.resize(V);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_random_cyclic_graph_77) {
  boost::mpi::communicator world;
  const int V = 77;

  std::vector<int> graph;

  std::vector<int> resMPI(V, 0);
  std::vector<int> resSEQ;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    graph = gnitienko_k_generate_func_mpi::generateGraph(V, gnitienko_k_generate_func_mpi::CYCLIC, 60);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
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
    resSEQ.resize(V);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
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

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_random_multigraph_30) {
  boost::mpi::communicator world;
  const int V = 30;

  std::vector<int> graph;

  std::vector<int> resMPI(V, 0);
  std::vector<int> resSEQ;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    graph = gnitienko_k_generate_func_mpi::generateGraph(V, gnitienko_k_generate_func_mpi::MULTIGRAPH, 40);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    resSEQ.resize(V);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    // Create Task
    gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    if (resSEQ == resMPI)
      ASSERT_EQ(resSEQ, resMPI);
    else
      GTEST_SKIP();
  }
}