// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/anikin_m_summ_of_different_symbols/include/ops_mpi.hpp"

TEST(SummDifSymMPI_parallel_perf_test, test_pipeline_run) {
  boost::mpi::communicator com;
  char str1[] = "kdfoew";
  char str2[] = "hfgeew";
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto testClassPar =
      std::make_shared<Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel>(taskDataPar);
  ASSERT_EQ(testClassPar->validation(), true);
  testClassPar->pre_processing();
  testClassPar->run();
  testClassPar->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(4, out[0]);
  }
}
TEST(SummDifSymMPI_parallel_perf_test, test_task_run) {
  boost::mpi::communicator com;
  char str1[] = "herwrwer";
  char str2[] = "httrrewr";
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create Task Data Parallel//
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto testClassPar =
      std::make_shared<Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel>(taskDataPar);
  ASSERT_EQ(testClassPar->validation(), true);
  testClassPar->pre_processing();
  testClassPar->run();
  testClassPar->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(5, out[0]);
  }
}