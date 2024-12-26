#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zolotareva_a_SLE_gradient_method/include/ops_mpi.hpp"

TEST(mpi_zolotareva_a_SLE_gradient_method_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  const uint32_t n = 1000;
  std::vector<double> A(n * n, 0);
  std::vector<double> b(n, 0);
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.push_back(n * n);
    taskDataPar->inputs_count.push_back(n);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskDataPar->outputs_count.push_back(x.size());
  }

  auto testMpiTaskParallel = std::make_shared<zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(n, taskDataPar->inputs_count[1]);
  }
}

TEST(mpi_zolotareva_a_SLE_gradient_method_perf_test, test_task_run) {
  boost::mpi::communicator world;
  const uint32_t n = 1000;
  std::vector<double> A(n * n, 0);
  std::vector<double> b(n, 0);
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.push_back(n * n);
    taskDataPar->inputs_count.push_back(n);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
    taskDataPar->outputs_count.push_back(x.size());
  }

  auto testMpiTaskParallel = std::make_shared<zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(n, taskDataPar->inputs_count[1]);
  }
}
