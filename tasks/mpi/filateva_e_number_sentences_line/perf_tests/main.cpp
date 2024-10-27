// Filateva Elizaveta Number_of_sentences_per_line
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filateva_e_number_sentences_line/include/ops_mpi.hpp"

TEST(filateva_e_number_sentences_line_mpi, test_pipeline_run) {
  int count = 22;
  boost::mpi::communicator world;
  std::string line = "Hello world.";
  std::vector<int> out(1, 0);
  // // Create TaskData
  for (int i = 0; i < count; i++) {
    line = line + line;
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto NumS = std::make_shared<filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel>(taskDataPar);
  ASSERT_EQ(NumS->validation(), true);
  NumS->pre_processing();
  NumS->run();
  NumS->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(NumS);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(4194304, out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, test_task_run) {
  int count = 22;
  boost::mpi::communicator world;
  std::string line = "Hello world.";
  std::vector<int> out(1, 0);
  // // Create TaskData
  for (int i = 0; i < count; i++) {
    line = line + line;
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto NumS = std::make_shared<filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel>(taskDataPar);
  ASSERT_EQ(NumS->validation(), true);
  NumS->pre_processing();
  NumS->run();
  NumS->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(NumS);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(4194304, out[0]);
  }
}