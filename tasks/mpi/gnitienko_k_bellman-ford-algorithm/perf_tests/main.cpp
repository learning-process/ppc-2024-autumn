#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gnitienko_k_bellman-ford-algorithm/include/ops_mpi.hpp"

namespace gnitienko_k_func_mpi {

const int INF = 1000000000;
const int MIN_WEIGHT = -1;
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
}  // namespace gnitienko_k_func_mpi

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int V = 1000;
  int E = 0;

  // Create data
  std::vector<int> graph;
  std::vector<int> resMPI;
  std::vector<int> resSEQ;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    graph = gnitienko_k_func_mpi::generateGraph(V);
    for (size_t i = 0; i < graph.size(); i++)
      if (graph[i] != 0) E++;
    resMPI.resize(V);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  auto testMpiTaskParallel = std::make_shared<gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  bool negative_cycle = false;
  if (testMpiTaskParallel->run() == false) negative_cycle = true;
  boost::mpi::broadcast(world, negative_cycle, 0);
  if (negative_cycle) GTEST_SKIP();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    resSEQ.resize(V);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    auto testTaskSequential = std::make_shared<gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq>(taskDataSeq);
    ASSERT_EQ(testTaskSequential->validation(), true);
    testTaskSequential->pre_processing();
    testTaskSequential->run();
    testTaskSequential->post_processing();
  }

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(resMPI, resSEQ);
  }
}

TEST(gnitienko_k_bellman_ford_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int V = 7000;
  const int E = 10000;

  // Create data
  std::vector<int> graph;
  std::vector<int> resMPI;
  std::vector<int> resSEQ;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    graph = gnitienko_k_func_mpi::generateGraph(V);
    resMPI.resize(V);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  auto testMpiTaskParallel = std::make_shared<gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  bool negative_cycle = false;
  if (testMpiTaskParallel->run() == false) negative_cycle = true;
  boost::mpi::broadcast(world, negative_cycle, 0);
  if (negative_cycle) GTEST_SKIP();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    resSEQ.resize(V);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
    taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    auto testTaskSequential = std::make_shared<gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq>(taskDataSeq);
    ASSERT_EQ(testTaskSequential->validation(), true);
    testTaskSequential->pre_processing();
    testTaskSequential->run();
    testTaskSequential->post_processing();
  }

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(resSEQ, resMPI);
  }
}
