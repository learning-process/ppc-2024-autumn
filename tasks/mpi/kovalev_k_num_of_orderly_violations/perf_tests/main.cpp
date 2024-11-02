// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_mpi, test_pipeline_run) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> g_vec;
  std::vector<size_t> g_num_viol(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  size_t length;
  const int alpha = 1;
  if (rank == 0) {
    length = 10;
    g_vec = std::vector<int>(length, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(g_vec.data()));
    taskDataPar->inputs_count.emplace_back(g_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(g_num_viol.data()));
    taskDataPar->outputs_count.emplace_back(g_num_viol.size());
  }
  auto testMpiParallel =
      std::make_shared<kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int>>(taskDataPar);
  ASSERT_EQ(testMpiParallel->validation(), true);
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    size_t res = 0;
    ASSERT_EQ(res, g_num_viol[0]);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
