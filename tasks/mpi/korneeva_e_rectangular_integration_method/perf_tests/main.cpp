#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/korneeva_e_rectangular_integration_method/include/ops_mpi.hpp"

namespace korneeva_e_rectangular_integration_method_mpi {
double test_func(std::vector<double> &args) { return args[0]; }
}  // namespace korneeva_e_rectangular_integration_method_mpi

TEST(korneeva_e_rectangular_integration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator mpi_comm;
  std::vector<std::pair<double, double>> limits(10, {-1000, 1000});
  korneeva_e_rectangular_integration_method_mpi::Function func =
      korneeva_e_rectangular_integration_method_mpi::test_func;
  std::vector<double> output(1);
  double epsilon = 1e-4;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (mpi_comm.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskData->inputs_count.emplace_back(limits.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskData->outputs_count.emplace_back(output.size());
  }

  auto testTaskParallel =
      std::make_shared<korneeva_e_rectangular_integration_method_mpi::RectangularIntegrationMPI>(taskData, func);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (mpi_comm.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(korneeva_e_rectangular_integration_method_mpi, test_task_run) {
  boost::mpi::communicator mpi_comm;
  std::vector<std::pair<double, double>> limits(10, {-1000, 1000});
  korneeva_e_rectangular_integration_method_mpi::Function func =
      korneeva_e_rectangular_integration_method_mpi::test_func;
  std::vector<double> output(1);
  double epsilon = 1e-4;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (mpi_comm.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskData->inputs_count.emplace_back(limits.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskData->outputs_count.emplace_back(output.size());
  }

  auto testTaskParallel =
      std::make_shared<korneeva_e_rectangular_integration_method_mpi::RectangularIntegrationMPI>(taskData, func);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (mpi_comm.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
