// Copyright 2023 Nesterov Alexander
#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method_mpi {

TEST(malyshev_v_conjugate_gradient_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> A = {{5, 2, 0}, {2, 5, 2}, {0, 2, 5}};
  std::vector<double> b = {1, 2, 3};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<double> res(b.size());

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&A));
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
    taskData->inputs_count.push_back(1);
    taskData->inputs_count.push_back(1);
    taskData->outputs_count.push_back(1);
  }
  auto task = std::make_shared<malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel>(taskData, A, b);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> A = {{5, 2, 0}, {2, 5, 2}, {0, 2, 5}};
  std::vector<double> b = {1, 2, 3};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<double> res(b.size());

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&A));
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
    taskData->inputs_count.push_back(1);
    taskData->inputs_count.push_back(1);
    taskData->outputs_count.push_back(1);
  }
  auto task = std::make_shared<malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel>(taskData, A, b);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}