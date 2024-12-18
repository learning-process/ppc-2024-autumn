// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korovin_n_line_topology/include/ops_mpi.hpp"

namespace korovin_n_line_topology_mpi {
std::vector<int> generate_rnd_vector(int size, int lower_bound = -500, int upper_bound = 500) {
  std::vector<int> v1(size);
  for (auto& num : v1) {
    num = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return v1;
}
}  // namespace korovin_n_line_topology_mpi

TEST(korovin_n_line_topology_mpi, test_task_run) {
  boost::mpi::communicator world;
  int n = 10000000;
  auto root = 0;
  auto dst = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(root);
  taskData->inputs_count.emplace_back(dst);
  taskData->inputs_count.emplace_back(n);

  std::vector<int> data;
  std::vector<int> received_data;
  std::vector<int> received_trajectory;

  if (world.rank() == root) {
    data = korovin_n_line_topology_mpi::generate_rnd_vector(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (root != dst) {
      world.send(dst, 0, data);
    }
  }

  if (world.rank() == dst) {
    int trajectory_size = dst - root + 1;

    if (root != dst) {
      world.recv(root, 0, data);
    }

    received_data.resize(n);
    received_trajectory.resize(trajectory_size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_trajectory.data()));
    taskData->outputs_count.emplace_back(received_trajectory.size());
  }

  auto testTask = std::make_shared<korovin_n_line_topology_mpi::TestMPITaskParallel>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == dst) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < n; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
    for (int i = 0; i < (int)received_trajectory.size(); i++) {
      ASSERT_EQ(received_trajectory[i], root + i);
    }
  }
}

TEST(korovin_n_line_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int n = 10000000;
  auto root = 0;
  auto dst = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(root);
  taskData->inputs_count.emplace_back(dst);
  taskData->inputs_count.emplace_back(n);

  std::vector<int> data;
  std::vector<int> received_data;
  std::vector<int> received_trajectory;

  if (world.rank() == root) {
    data = korovin_n_line_topology_mpi::generate_rnd_vector(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (root != dst) {
      world.send(dst, 0, data);
    }
  }

  if (world.rank() == dst) {
    int trajectory_size = dst - root + 1;

    if (root != dst) {
      world.recv(root, 0, data);
    }

    received_data.resize(n);
    received_trajectory.resize(trajectory_size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_trajectory.data()));
    taskData->outputs_count.emplace_back(received_trajectory.size());
  }

  auto testTask = std::make_shared<korovin_n_line_topology_mpi::TestMPITaskParallel>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == dst) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < n; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
    for (int i = 0; i < (int)received_trajectory.size(); i++) {
      ASSERT_EQ(received_trajectory[i], root + i);
    }
  }
}
