#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/solovev_a_star_topology/include/ops_mpi.hpp"

TEST(solovev_a_star_topology_mpi_perf_test, test_pipeline_run) {
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(10000);
  std::vector<int> output(10000, 0);
  boost::mpi::communicator world;
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }
  auto testMpiTaskParallel = std::make_shared<solovev_a_star_topology_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      ASSERT_EQ(output, input);
    }
  }
}

TEST(solovev_a_star_topology_mpi_perf_test, test_task_run) {
  std::vector<int> input = solovev_a_star_topology_mpi::generate_random_vector(10000);
  std::vector<int> output(10000, 0);
  boost::mpi::communicator world;
  std::vector<int> order(world.size() + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(order.data()));
    taskDataPar->outputs_count.emplace_back(order.size());
  }
  auto testMpiTaskParallel = std::make_shared<solovev_a_star_topology_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.size() == 1) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      ASSERT_EQ(output, input);
    }
  }
}
