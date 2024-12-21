// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korotin_e_multidimentional_integrals_monte_carlo/include/ops_mpi.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_mpi {

double test_func(double *x, int x_size) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_mpi

TEST(korotin_e_multidimentional_integrals_monte_carlo, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> left_border(3);
  std::vector<double> right_border(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  std::vector<double (*)(double *, int)> F(1, &korotin_e_multidimentional_integrals_monte_carlo_mpi::test_func);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(F.data()));
  taskDataPar->inputs_count.emplace_back(F.size());
  if (world.rank() == 0) {
    left_border[0] = left_border[1] = left_border[2] = 0.0;
    right_border[0] = right_border[1] = right_border[2] = 2.0;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(left_border.data()));
    taskDataPar->inputs_count.emplace_back(left_border.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_border.data()));
    taskDataPar->inputs_count.emplace_back(right_border.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
    taskDataPar->inputs_count.emplace_back(N.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  double err = testMpiTaskParallel->possible_error();

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double ref = 32.0;
    bool ans = (std::abs(res[0] - ref) < err);

    ASSERT_EQ(ans, true);
  }
}

TEST(korotin_e_multidimentional_integrals_monte_carlo, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> left_border(3);
  std::vector<double> right_border(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  std::vector<double (*)(double *, int)> F(1, &korotin_e_multidimentional_integrals_monte_carlo_mpi::test_func);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(F.data()));
  taskDataPar->inputs_count.emplace_back(F.size());

  if (world.rank() == 0) {
    left_border[0] = left_border[1] = left_border[2] = 0.0;
    right_border[0] = right_border[1] = right_border[2] = 2.0;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(left_border.data()));
    taskDataPar->inputs_count.emplace_back(left_border.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_border.data()));
    taskDataPar->inputs_count.emplace_back(right_border.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
    taskDataPar->inputs_count.emplace_back(N.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  double err = testMpiTaskParallel->possible_error();

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double ref = 32.0;
    bool ans = (std::abs(res[0] - ref) < err);

    ASSERT_EQ(ans, true);
  }
}
