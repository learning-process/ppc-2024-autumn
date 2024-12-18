#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tyurin_m_count_sentences_in_string/include/ops_mpi.hpp"

const size_t count_strings = 10000;

TEST(tyurin_m_count_sentences_in_string_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);
  std::string str;

  if (world.rank() == 0) {
    str = "This is the first sentence. And this is the second! Finally, the third?";
    input_str.resize(str.size() * count_strings);
    for (size_t i = 0; i < count_strings; i++) {
      std::copy(str.begin(), str.end(), input_str.begin() + i * str.size());
    }
    global_count[0] = 0;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
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
    ASSERT_EQ(30000, global_count[0]);
  }
}

TEST(tyurin_m_count_sentences_in_string_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);
  std::string str;

  if (world.rank() == 0) {
    str = "This is the first sentence. And this is the second! Finally, the third?";
    input_str.resize(str.size() * count_strings);
    for (size_t i = 0; i < count_strings; i++) {
      std::copy(str.begin(), str.end(), input_str.begin() + i * str.size());
    }
    global_count[0] = 0;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
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
    ASSERT_EQ(30000, global_count[0]);
  }
}