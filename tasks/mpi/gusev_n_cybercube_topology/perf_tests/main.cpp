#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gusev_n_cybercube_topology/include/ops_mpi.hpp"

namespace gusev_n_cybercube_topology_mpi {

TEST(gusev_n_cybercube_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) && (world.size() >= 2)) {
    std::vector<uint8_t> input_data(1024, 1);
    std::vector<uint8_t> output_data(1024);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.push_back(input_data.data());
    task_data->inputs_count.push_back(input_data.size());
    task_data->outputs.push_back(output_data.data());
    task_data->outputs_count.push_back(output_data.size());

    CybercubeTopologyParallel task(task_data);
    ASSERT_TRUE(task.validation());

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    perfAttr->num_running = 10;

    ppc::core::Perf perfAnalyzer(std::make_shared<CybercubeTopologyParallel>(task_data));

    for (int i = 0; i < perfAttr->num_running; ++i) {
      perfAnalyzer.pipeline_run(perfAttr, perfResults);
    }

    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(gusev_n_cybercube_topology_mpi, test_task_run) {
  boost::mpi::communicator world;

  if ((std::ceil(std::log2(world.size())) == std::log2(world.size())) && (world.size() >= 2)) {
    std::vector<uint8_t> input_data(1024, 1);
    std::vector<uint8_t> output_data(1024);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.push_back(input_data.data());
    task_data->inputs_count.push_back(input_data.size());
    task_data->outputs.push_back(output_data.data());
    task_data->outputs_count.push_back(output_data.size());

    CybercubeTopologyParallel task(task_data);
    ASSERT_TRUE(task.validation());

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    perfAttr->num_running = 10;

    ppc::core::Perf perfAnalyzer(std::make_shared<CybercubeTopologyParallel>(task_data));

    for (int i = 0; i < perfAttr->num_running; ++i) {
      perfAnalyzer.task_run(perfAttr, perfResults);
    }

    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

}  // namespace gusev_n_cybercube_topology_mpi
