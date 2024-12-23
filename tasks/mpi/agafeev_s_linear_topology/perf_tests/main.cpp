#include <gtest/gtest.h>

#include "boost/mpi/communicator.hpp"
#include "boost/mpi/timer.hpp"
#include "core/perf/include/perf.hpp"
#include "mpi/agafeev_s_linear_topology/include/lintop_mpi.hpp"

TEST(agafeev_s_linear_topology, test_pipeline_run) {
  boost::mpi::communicator world;
  int sender = 0;
  int receiver = 1;
  std::vector<int> right_route = agafeev_s_linear_topology::calculating_Route(sender, receiver);
  bool out = false;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sender));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&receiver));

  if (world.rank() == sender) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_route.data()));
    taskData->inputs_count.emplace_back(right_route.size());
  } else {
    if (world.rank() == receiver) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
      taskData->outputs_count.emplace_back(1);
    }
  }

  auto testTaskMpi = std::make_shared<agafeev_s_linear_topology::LinearTopology>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMpi);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == receiver) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(out);
  }
}

TEST(agafeev_s_linear_topology, test_task_run) {
  boost::mpi::communicator world;
  int sender = 0;
  int receiver = 1;
  std::vector<int> right_route = agafeev_s_linear_topology::calculating_Route(sender, receiver);
  bool out = false;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sender));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&receiver));

  if (world.rank() == sender) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_route.data()));
    taskData->inputs_count.emplace_back(right_route.size());
  } else {
    if (world.rank() == receiver) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
      taskData->outputs_count.emplace_back(1);
    }
  }

  auto testTaskMpi = std::make_shared<agafeev_s_linear_topology::LinearTopology>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMpi);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == receiver) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(out);
  }
}
