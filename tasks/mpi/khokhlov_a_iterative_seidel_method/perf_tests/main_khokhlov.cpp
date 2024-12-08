#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/khokhlov_a_iterative_seidel_method/include/ops_mpi_khokhlov.hpp"

TEST(khokhlov_a_iterative_seidel_method_mpi, test_pipline_run) {
  boost::mpi::communicator world;
  const int n = 800;
  const int maxiter = 800;
  const double eps = 1e-3;

  // create data
  std::vector<double> A(n * n, 0.0);
  std::vector<double> b(n, 0.0);
  std::vector<double> result(n, 0.0);
  khokhlov_a_iterative_seidel_method_mpi::getRandomSLAU(A, b, n);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  auto testTaskMpi = std::make_shared<khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi>(taskdataMpi);
  ASSERT_EQ(testTaskMpi->validation(), true);
  testTaskMpi->pre_processing();
  testTaskMpi->run();
  testTaskMpi->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMpi);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int n = 800;
  const int maxiter = 800;
  const double eps = 1e-3;

  // create data
  std::vector<double> A(n * n, 0.0);
  std::vector<double> b(n, 0.0);
  std::vector<double> result(n, 0.0);
  khokhlov_a_iterative_seidel_method_mpi::getRandomSLAU(A, b, n);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  auto testTaskMpi = std::make_shared<khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi>(taskdataMpi);
  ASSERT_EQ(testTaskMpi->validation(), true);
  testTaskMpi->pre_processing();
  testTaskMpi->run();
  testTaskMpi->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMpi);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}