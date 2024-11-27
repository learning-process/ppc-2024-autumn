#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/alputov_i_topology_hypercube/include/ops_mpi.hpp"

namespace alputov_i_topology_hypercube_mpi {

std::vector<int> removeNegativeOnesFromEnd(std::vector<int> vector) {
  auto it = std::find_if_not(vector.rbegin(), vector.rend(), [](int value) { return value == -1; });
  vector.erase(it.base(), vector.end());
  return vector;
}

}  // namespace alputov_i_topology_hypercube_mpi

TEST(alputov_i_topology_hypercube_mpi, PipelineExecutionTest) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if (!(communicatorSize & (communicatorSize - 1)) && communicatorSize > 0) {
    std::vector<int> inputData{1337, 0};
    std::vector<int> expectedRoute{0};
    std::vector<int> outputData(1, 0);

    std::vector<int> actualRoute;
    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    auto hypercubeRouter = std::make_shared<alputov_i_topology_hypercube_mpi::HypercubeRouterMPI>(taskData);
    ASSERT_EQ(hypercubeRouter->validation(), true);
    hypercubeRouter->pre_processing();
    hypercubeRouter->run();
    hypercubeRouter->post_processing();

    auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
    performanceAttributes->num_running = 10;
    const boost::mpi::timer timer;
    performanceAttributes->current_timer = [&] { return timer.elapsed(); };

    auto performanceResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(hypercubeRouter);
    perfAnalyzer->pipeline_run(performanceAttributes, performanceResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(performanceResults);
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}

TEST(alputov_i_topology_hypercube_mpi, TaskExecutionTest) {
  boost::mpi::communicator world;
  size_t communicatorSize = world.size();
  if (!(communicatorSize & (communicatorSize - 1)) && communicatorSize > 0) {
    std::vector<int> inputData{1337, 0};
    std::vector<int> expectedRoute{0};
    std::vector<int> outputData(1, 0);

    std::vector<int> actualRoute;
    size_t routeSize = std::log2(communicatorSize) + 1;
    actualRoute.reserve(routeSize);
    for (size_t i = 0; i < routeSize; ++i) {
      actualRoute.push_back(-1);
    }

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
      taskData->inputs_count.emplace_back(inputData.size());
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
      taskData->outputs_count.emplace_back(outputData.size());
      taskData->outputs_count.emplace_back(actualRoute.size());
    }
    auto hypercubeRouter = std::make_shared<alputov_i_topology_hypercube_mpi::HypercubeRouterMPI>(taskData);
    ASSERT_EQ(hypercubeRouter->validation(), true);
    hypercubeRouter->pre_processing();
    hypercubeRouter->run();
    hypercubeRouter->post_processing();

    auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
    performanceAttributes->num_running = 10;
    const boost::mpi::timer timer;
    performanceAttributes->current_timer = [&] { return timer.elapsed(); };

    auto performanceResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(hypercubeRouter);
    perfAnalyzer->pipeline_run(performanceAttributes, performanceResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(performanceResults);
      actualRoute = alputov_i_topology_hypercube_mpi::removeNegativeOnesFromEnd(actualRoute);
      ASSERT_EQ(outputData[0], inputData[0]);
      ASSERT_EQ(actualRoute, expectedRoute);
    }
  }
}
