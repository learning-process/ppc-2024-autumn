#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/smirnov_i_tape_splitting_A/include/ops_mpi.hpp"

TEST(smirnov_i_tape_splitting_A_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 700;
  int n_a = 800;
  int m_b = 800;
  int n_b = 300;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = new double[m_a * n_a]();
    B = new double[m_b * n_b]();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);

    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }

  auto testMpiTaskParallel = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskParallel>(taskDataPar);
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
    auto* expected_res = new double[m_a * n_b]();
    for (int i = 0; i < m_a * n_b; i++) {
      ASSERT_NEAR(expected_res[i], res[i], 1e-6);
    }
    delete[] expected_res;
    delete[] res;
    delete[] A;
    delete[] B;
  }
}

TEST(smirnov_i_tape_splitting_A_mpi, test_task_run) {
  boost::mpi::communicator world;
  double* A = nullptr;
  double* B = nullptr;
  double* res = nullptr;
  int m_a = 700;
  int n_a = 800;
  int m_b = 800;
  int n_b = 300;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = new double[m_a * n_a]();
    B = new double[m_b * n_b]();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

    taskDataPar->inputs_count.emplace_back(m_a);
    taskDataPar->inputs_count.emplace_back(n_a);
    taskDataPar->inputs_count.emplace_back(m_b);
    taskDataPar->inputs_count.emplace_back(n_b);

    res = new double[m_a * n_b];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
    taskDataPar->outputs_count.emplace_back(m_a);
    taskDataPar->outputs_count.emplace_back(n_b);
  }

  auto testMpiTaskParallel = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskParallel>(taskDataPar);
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
    auto* expected_res = new double[m_a * n_b]();
    for (int i = 0; i < m_a * n_b; i++) {
      ASSERT_NEAR(expected_res[i], res[i], 1e-6);
    }
    delete[] expected_res;
    delete[] res;
    delete[] A;
    delete[] B;
  }
}