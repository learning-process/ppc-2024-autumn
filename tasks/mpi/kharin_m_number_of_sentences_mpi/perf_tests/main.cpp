#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kharin_m_number_of_sentences_mpi/include/ops_mpi.hpp"

TEST(mpi_kharin_m_sentence_count_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> sentence_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  input_text = "This is a long text with many sentences. ";
  for (int i = 0; i < 10000000; i++) {
    input_text += "Sentence " + std::to_string(i + 1) + ". ";
  }

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataPar->inputs_count.emplace_back(input_text.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  auto testMpiTaskParallel = std::make_shared<kharin_m_number_of_sentences_mpi::CountSentencesParallel>(taskDataPar);
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
    // Проверяем результат (должно быть 10000001 предложение)
    ASSERT_EQ(10000001, sentence_count[0]);
  }
}

TEST(mpi_kharin_m_sentence_count_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  input_text = "This is a long text with many sentences. ";
  for (int i = 0; i < 10000000; i++) {
    input_text += "Sentence " + std::to_string(i + 1) + ". ";
  }

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataPar->inputs_count.emplace_back(input_text.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  auto testMpiTaskParallel = std::make_shared<kharin_m_number_of_sentences_mpi::CountSentencesParallel>(taskDataPar);
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
    // Проверяем результат (должно быть 10000001 предложение)
    ASSERT_EQ(10000001, sentence_count[0]);
  }
}
