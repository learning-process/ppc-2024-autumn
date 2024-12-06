#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/src/ops_mpi.cpp"

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_pipeline_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(3);

  auto seidelTask = std::make_shared<nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI>(taskData);

  ASSERT_TRUE(seidelTask->validation()) << "Validation failed for valid input";

  seidelTask->pre_processing();
  seidelTask->run();
  seidelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seidelTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_task_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(3);

  auto seidelTask = std::make_shared<nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI>(taskData);

  ASSERT_TRUE(seidelTask->validation()) << "Validation failed for valid input";

  seidelTask->pre_processing();
  seidelTask->run();
  seidelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seidelTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}