#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kholin_k_iterative_methods_Seidel/include/ops_mpi.hpp"

TEST(kholin_k_iterative_methods_Seidel_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int ProcRank = 0;
  const size_t count_rows = 2500;
  const size_t count_colls = 2500;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  float* in = new float[count_rows * count_colls];
  float* out = new float[count_rows];
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in, count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out));
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
  const size_t count_rows = 3000;
  const size_t count_colls = 3000;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  float* in = new float[count_rows * count_colls];
  float* out = new float[count_rows];
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in, count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out));
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

// int main(int argc, char** argv) {
//   boost::mpi::environment env(argc, argv);
//   boost::mpi::communicator world;
//   ::testing::InitGoogleTest(&argc, argv);
//   ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
//   if (world.rank() != 0) {
//     delete listeners.Release(listeners.default_result_printer());
//   }
//   return RUN_ALL_TESTS();
// }
