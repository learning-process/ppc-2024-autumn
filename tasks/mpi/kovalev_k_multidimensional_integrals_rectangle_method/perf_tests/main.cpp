#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <ctime>
#include <numbers>

#include "core/perf/include/perf.hpp"
#include "mpi/kovalev_k_multidimensional_integrals_rectangle_method/include/header.hpp"

namespace kovalev_k_multidimensional_integrals_rectangle_method_mpi {
double f1cos(std::vector<double> &arguments) { return std::cos(arguments.at(0)); }
double f2advanced(std::vector<double> &arguments) { return std::tan(arguments.at(0)) * std::atan(arguments.at(1)); }
}  // namespace kovalev_k_multidimensional_integrals_rectangle_method_mpi

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, test_pipeline_run) {
  const size_t dim = 1;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = -0.5 * std::numbers::pi;
  lims[0].second = 0.5 * std::numbers::pi;
  double h = 0.0005;
  double eps = 1e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(lims.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  auto testMpiParallel = std::make_shared<
      kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar>(
      tmpPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f1cos);
  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(2.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, test_task_run) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = lims[1].first = 0.0;
  lims[0].second = lims[1].second = 1.5;
  double h = 0.0005;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(lims.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  auto testMpiParallel = std::make_shared<
      kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar>(
      taskDataPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f2advanced);
  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(2.34381088006031, out[0], eps);
  }
}
