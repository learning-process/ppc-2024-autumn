#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sharamygina_i_line_topology/include/ops_mpi.h"

TEST(sharamygina_i_line_topology_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 10000000;
  auto sendler = 0;
  auto recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data(size, 0);
  std::iota(data.begin(), data.end(), 0);
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (sendler != recipient) {
      world.send(recipient, 0, data);
    }
  }

  if (world.rank() == recipient) {
    if (sendler != recipient) {
      world.recv(sendler, 0, data);
    }

    received_data.resize(size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  auto testTask = std::make_shared<sharamygina_i_line_topology_mpi::line_topology_mpi>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == recipient) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
  }
}

TEST(sharamygina_i_line_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 5000000;
  auto sendler = 0;
  auto recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data(size, 0);
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (sendler != recipient) {
      world.send(recipient, 0, data);
    }
  }

  if (world.rank() == recipient) {
    if (sendler != recipient) {
      world.recv(sendler, 0, data);
    }

    received_data.resize(size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  auto testTask = std::make_shared<sharamygina_i_line_topology_mpi::line_topology_mpi>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 9;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == recipient) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
  }
}