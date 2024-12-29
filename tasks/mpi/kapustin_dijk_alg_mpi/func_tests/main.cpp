#include <gtest/gtest.h>

#include "mpi/kapustin_dijk_alg_mpi/include/mpi_alg.hpp"
namespace generateGraph {
std::vector<int> generateGraph(int V, int E) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> graph(V * V, 0);
  std::set<std::pair<int, int>> edges;

  std::uniform_int_distribution<> dist_vertex(0, V - 1);
  std::uniform_int_distribution<> dist_weight(1, 10);

  int edgeCount = 0;
  while (edgeCount < E) {
    int u = dist_vertex(gen);
    int v = dist_vertex(gen);
    if (u != v && edges.find({u, v}) == edges.end()) {
      int weight = dist_weight(gen);
      graph[u * V + v] = weight;
      graph[v * V + u] = weight;
      edges.insert({u, v});
      edges.insert({v, u});
      edgeCount++;
    }
  }
  return graph;
}
}  // namespace generateGraph
TEST(kapustin_dijkstras_algorithm_mpi, random_graph_with_large_data_input) {
  boost::mpi::communicator world;
  const int V = 1000;
  const int E = 2000;
  std::vector<int> graph = generateGraph::generateGraph(V, E);
  std::vector<int> resSEQ(V, 0);
  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());
  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
  taskDataSeq->outputs_count.emplace_back(resSEQ.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, resSEQ);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, random_graph) {
  boost::mpi::communicator world;
  const int V = 5;
  const int E = 8;
  std::vector<int> graph = generateGraph::generateGraph(V, E);
  std::vector<int> resSEQ(V, 0);
  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());
  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
  taskDataSeq->outputs_count.emplace_back(resSEQ.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, resSEQ);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, eq_seq_mpi) {
  boost::mpi::communicator world;
  const int V = 5;
  const int E = 8;
  std::vector<int> graph = {0, 10, 0, 0, 0, 10, 0, 5, 0, 0, 0, 5, 0, 20, 0, 0, 0, 20, 0, 1, 0, 0, 0, 1, 0};
  std::vector<int> resSEQ(V, 0);
  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
  taskDataSeq->outputs_count.emplace_back(resSEQ.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, resSEQ);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, simple_2nd) {
  boost::mpi::communicator world;
  const int V = 6;
  const int E = 8;

  std::vector<int> graph = {0, 7,  9,  0, 0, 14, 7, 0, 10, 15, 0, 0, 9,  10, 0, 11, 0, 2,
                            0, 15, 11, 0, 6, 0,  0, 0, 0,  6,  0, 9, 14, 0,  2, 0,  9, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 7, 9, 20, 20, 11};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());
  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, expected_res);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, empty_graph) {
  boost::mpi::communicator world;
  const int V = 4;
  const int E = 0;

  std::vector<int> graph = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, INT_MAX, INT_MAX, INT_MAX};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, expected_res);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, test_single_edge_graph) {
  boost::mpi::communicator world;
  const int V = 2;
  const int E = 1;

  std::vector<int> graph = {0, 5, 5, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 5};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, expected_res);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, test_cyclic_graph) {
  boost::mpi::communicator world;
  const int V = 4;
  const int E = 4;

  std::vector<int> graph = {0, 1, 0, 10, 1, 0, 2, 0, 0, 2, 0, 3, 10, 0, 3, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 1, 3, 6};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, expected_res);
  }
}
TEST(kapustin_dijkstras_algorithm_mpi, test_disconnected_graph) {
  boost::mpi::communicator world;
  const int V = 4;
  const int E = 2;

  std::vector<int> graph = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0};
  std::vector<int> resMPI(V, 0);
  std::vector<int> expected_res = {0, 1, INT_MAX, INT_MAX};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
  taskDataPar->outputs_count.emplace_back(resMPI.size());

  kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(resMPI, expected_res);
  }
}