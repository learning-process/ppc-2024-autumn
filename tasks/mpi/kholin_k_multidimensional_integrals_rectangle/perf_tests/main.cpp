#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kholin_k_multidimensional_integrals_rectangle/include/ops_mpi.hpp"

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return std::cos(f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]) *
           (1 + f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]);
  };
  std::vector<double> in_lower_limits{-1, 2, -3};
  std::vector<double> in_upper_limits{8, 8, 2};
  double epsilon = 0.1;
  int n = 1;
  std::vector<double> out_I(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel>(taskDataPar, f);
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
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, test_task_run) {
  boost::mpi::communicator world;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return std::cos(f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]) *
           (1 + f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]);
  };
  std::vector<double> in_lower_limits{-1, 2, -3};
  std::vector<double> in_upper_limits{8, 8, 2};
  double epsilon = 0.1;
  int n = 1;
  std::vector<double> out_I(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel>(taskDataPar, f);
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
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}