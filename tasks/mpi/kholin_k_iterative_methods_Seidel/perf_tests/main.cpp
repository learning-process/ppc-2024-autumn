#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <mpi/kholin_k_iterative_methods_Seidel/src/ops_mpi.cpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kholin_k_iterative_methods_Seidel/include/ops_mpi.hpp"

TEST(kholin_k_iterative_methods_Seidel_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int ProcRank = 0;
  const size_t count_rows = 1000;
  const size_t count_colls = 1000;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    float p1 = -(10.0f * 10.0f * 10.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  auto testMpiTaskParallel =
      std::make_shared<kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel>(taskDataPar, op);
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

TEST(kholin_k_iterative_methods_Seidel_mpi, test_task_run) {
  boost::mpi::communicator world;
  int ProcRank = 0;
  const size_t count_rows = 1000;
  const size_t count_colls = 1000;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    float p1 = -(10.0f * 10.0f * 10.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  auto testMpiTaskParallel =
      std::make_shared<kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel>(taskDataPar, op);
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