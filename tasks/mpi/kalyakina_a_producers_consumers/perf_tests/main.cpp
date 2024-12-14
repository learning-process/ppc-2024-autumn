// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kalyakina_a_producers_consumers/include/ops_mpi.hpp"

TEST(kalyakina_a_producers_consumers_mpi, ProducersConcumers_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2, 1), std::max(world.size() / 2, 1), world.size() * 2};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto TaskParallel =
      std::make_shared<kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel>(taskDataPar);
  ASSERT_EQ(TaskParallel->validation(), true);
  TaskParallel->pre_processing();
  TaskParallel->run();
  TaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(out[0], 0);
  }
}

TEST(kalyakina_a_producers_consumers_mpi, ProducersConcumers_task_run) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2, 1), std::max(world.size() / 2, 1), world.size() * 2};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto TaskParallel =
      std::make_shared<kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel>(taskDataPar);
  ASSERT_EQ(TaskParallel->validation(), true);
  TaskParallel->pre_processing();
  TaskParallel->run();
  TaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(out[0], 0);
  }
}