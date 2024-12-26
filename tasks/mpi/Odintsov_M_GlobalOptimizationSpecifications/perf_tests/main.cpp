#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/Odintsov_M_GlobalOptimizationSpecifications/include/ops_mpi.hpp"

TEST(Odintsov_m_OptimPar_MPI_perf_tests, test_pipeline_run) {
  boost::mpi::communicator com;
  // Create data
  double step = 0.3;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {5, 5};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1,
                                    2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
  std::vector<double> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(12);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel>(
          taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(46.777778, out[0], 0.0001);
  }
}

TEST(Odintsov_m_OptimPar_MPI_perf_tests, test_task_run) {
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {5, 5};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1,
                                    2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
  std::vector<double> out(1, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(12);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto testClassPar =
      std::make_shared<Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel>(
          taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(46.777778, out[0], 0.0001);
  }
}