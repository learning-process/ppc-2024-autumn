#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shkurinskaya_e_count_sentences/include/ops_mpi.hpp"

TEST(shkurinskaya_e_count_sentences_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_text = "This is another test. I love testing stuff! Don't you like it too?";
    for (int i = 0; i < 999999; i++) {
      input_text += "Let's do it " + std::to_string(i + 1) + " time!";
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel = std::make_shared<shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(1000002, global_result[0]);
  }
}

TEST(shkurinskaya_e_count_sentences_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_text = "This is another test. I love testing stuff! Don't you like it too?";
    for (int i = 0; i < 999999; i++) {
      input_text += "Let's do it " + std::to_string(i + 1) + " time!";
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel = std::make_shared<shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(1000002, global_result[0]);
  }
}
