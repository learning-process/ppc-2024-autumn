// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rysev_m_gypercube/include/ops_mpi.hpp"

TEST(rysev_m_gypercube, test_pipeline_run) {
  boost::mpi::communicator world;

  if ((world.size() & (world.size() - 1)) != 0) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 0;
  int _target = 1;
  int out = -1;
  std::vector<int> out_path(std::log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }

  auto testParTask = std::make_shared<rysev_m_gypercube::GyperCube>(taskDataPar);
  ASSERT_TRUE(testParTask->validation());
  testParTask->pre_processing();
  testParTask->run();
  testParTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == _target) {
    out_path.erase(std::remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(0, 0, out);
    world.send(0, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{0, 1};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(rysev_m_gypercube, test_task_run) {
  boost::mpi::communicator world;

  if ((world.size() & (world.size() - 1)) != 0) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 0;
  int _target = 1;
  int out = -1;
  std::vector<int> out_path(std::log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }

  auto testParTask = std::make_shared<rysev_m_gypercube::GyperCube>(taskDataPar);
  ASSERT_TRUE(testParTask->validation());
  testParTask->pre_processing();
  testParTask->run();
  testParTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == _target) {
    out_path.erase(std::remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(0, 0, out);
    world.send(0, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{0, 1};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}