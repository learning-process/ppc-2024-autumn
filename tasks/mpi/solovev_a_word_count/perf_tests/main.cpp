#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/solovev_a_word_count/include/ops_mpi.hpp"

std::vector<char> create_text(int quan_words) {
  std::vector<char> res;
  std::string word = "word ";
  std::string last = "word.";
  for (int i = 0; i < quan_words - 1; i++)
    for (unsigned long int symbol = 0; symbol < word.length(); symbol++) {
      res.push_back(word[symbol]);
    }
  for (unsigned long int symbol = 0; symbol < last.length(); symbol++) {
    res.push_back(last[symbol]);
  }
  return res;
}

std::vector<char> input_text = create_text(60000);

TEST(solovev_a_word_count_mpi_perf_test, test_pipeline_run) {
  std::vector<char> input = input_text;
  std::vector<int> out(1, 0);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  auto testMpiTaskParallel = std::make_shared<solovev_a_word_count_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(out[0], 60000);
  }
}

TEST(solovev_a_word_count_mpi_perf_test, test_task_run) {
  std::vector<char> input = input_text;
  std::vector<int> out(1, 0);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  auto testMpiTaskParallel = std::make_shared<solovev_a_word_count_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(out[0], 60000);
  }
}
