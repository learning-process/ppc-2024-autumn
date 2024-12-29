#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/milovankin_m_hypercube_topology/include/ops_mpi.hpp"

TEST(milovankin_m_hypercube_topology, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;  // tests are designed for 4+ processes

  const std::string data_input(16381, 'x');
  int dest = world.size() - 1;
  milovankin_m_hypercube_topology::Hypercube::DataIn data_in_struct(data_input, dest);
  milovankin_m_hypercube_topology::Hypercube::DataIn data_out_struct;

  std::vector<int> route_expected = milovankin_m_hypercube_topology::Hypercube::calculate_route(dest);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<milovankin_m_hypercube_topology::Hypercube>(taskDataPar);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Assert
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(data_out_struct.data, data_in_struct.data);
    ASSERT_EQ(data_out_struct.route, route_expected);
  }
}

TEST(milovankin_m_hypercube_topology, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;  // tests are designed for 4+ processes

  const std::string data_input(16381, 'x');
  int dest = world.size() - 1;
  milovankin_m_hypercube_topology::Hypercube::DataIn data_in_struct(data_input, dest);
  milovankin_m_hypercube_topology::Hypercube::DataIn data_out_struct;

  std::vector<int> route_expected = milovankin_m_hypercube_topology::Hypercube::calculate_route(dest);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<milovankin_m_hypercube_topology::Hypercube>(taskDataPar);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->task_run(perfAttr, perfResults);

  // Assert
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(data_out_struct.data, data_in_struct.data);
    ASSERT_EQ(data_out_struct.route, route_expected);
  }
}
