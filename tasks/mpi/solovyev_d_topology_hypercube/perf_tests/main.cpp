#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/solovyev_d_topology_hypercube/include/header.hpp"

namespace solovyev_d_topology_hypercube_mpi {
std::vector<int> removeTrailing(std::vector<int> vec) {
  vec.erase(std::remove(vec.begin(), vec.end(), -1), vec.end());
  return vec;
}
}  // namespace solovyev_d_topology_hypercube_mpi

TEST(solovyev_d_topology_hypercube_mpi, run_pipeline) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) >= 0)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 0};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()) + 1, -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    auto TopologyHypercubeMPI = std::make_shared<solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI>(taskDataPar);
    ASSERT_EQ(TopologyHypercubeMPI->validation(), true);
    TopologyHypercubeMPI->pre_processing();
    TopologyHypercubeMPI->run();
    TopologyHypercubeMPI->post_processing();

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TopologyHypercubeMPI);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      outputPath = solovyev_d_topology_hypercube_mpi::removeTrailing(outputPath);
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(solovyev_d_topology_hypercube_mpi, run_task) {
  boost::mpi::communicator world;
  // checking that number of processes is power of 2 and minimum number of processes required for current test
  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) and (std::log2(world.size()) >= 0)) {
    // input{<data>,<number of destination process>}
    std::vector<int> input{1024, 0};
    // expectedPath{<expected path of data transfer between processes>}
    std::vector<int> expectedPath{0};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(std::log2(world.size()) + 1, -1);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPath.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs_count.emplace_back(outputPath.size());
    }
    auto TopologyHypercubeMPI = std::make_shared<solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI>(taskDataPar);
    ASSERT_EQ(TopologyHypercubeMPI->validation(), true);
    TopologyHypercubeMPI->pre_processing();
    TopologyHypercubeMPI->run();
    TopologyHypercubeMPI->post_processing();

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TopologyHypercubeMPI);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      outputPath = solovyev_d_topology_hypercube_mpi::removeTrailing(outputPath);
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

int main(int argc, char* argv[]) {
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}