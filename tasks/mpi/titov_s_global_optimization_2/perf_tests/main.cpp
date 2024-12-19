// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/titov_s_global_optimization_2/include/ops_mpi.hpp"

TEST(titov_s_global_optimization_2_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x * p.x + p.y * p.y; };
  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.x; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  auto MPIOptimizationParallel =
      std::make_shared<titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel>(taskDataPar);
  ASSERT_EQ(MPIOptimizationParallel->validation(), true);
  MPIOptimizationParallel->pre_processing();
  MPIOptimizationParallel->run();
  MPIOptimizationParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIOptimizationParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(outPar[0].x, 0.0, 0.1);
  }
}

TEST(titov_s_global_optimization_2_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x * p.x + p.y * p.y; };
  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.x; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  auto MPIOptimizationParallel =
      std::make_shared<titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel>(taskDataPar);
  ASSERT_EQ(MPIOptimizationParallel->validation(), true);
  MPIOptimizationParallel->pre_processing();
  MPIOptimizationParallel->run();
  MPIOptimizationParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIOptimizationParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(outPar[0].x, 0.0, 0.1);
  }
}
