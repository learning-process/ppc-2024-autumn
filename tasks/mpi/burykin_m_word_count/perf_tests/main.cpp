#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "core/perf/include/perf.hpp"
#include "mpi/burykin_m_word_count/include/ops_mpi.hpp"

using namespace ppc::core;
using namespace burykin_m_word_count;

TEST(WordCountMPI, PipelineRunPerformance) {
  boost::mpi::communicator world;
  std::string input = "This is a sample text to test the word counting functionality.";
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskData->inputs_count.emplace_back(input_data.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskData->outputs_count.emplace_back(output_data.size());
  }

  auto task = std::make_shared<TestTaskParallel>(taskData);
  Perf perfAnalyzer(task);

  auto perfAttr = std::make_shared<PerfAttr>();
  perfAttr->num_running = 1000;
  perfAttr->current_timer = []() -> double {
    return static_cast<double>(std::chrono::steady_clock::now().time_since_epoch().count()) * 1e-9;
  };

  auto perfResults = std::make_shared<PerfResults>();

  perfAnalyzer.pipeline_run(perfAttr, perfResults);
  Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    ASSERT_EQ(output_data[0], 11);
  }
}

TEST(WordCountMPI, TaskRunPerformance) {
  boost::mpi::communicator world;
  std::string input = "Another example sentence to evaluate the performance of the word counting task.";
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskData->inputs_count.emplace_back(input_data.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskData->outputs_count.emplace_back(output_data.size());
  }

  auto task = std::make_shared<TestTaskParallel>(taskData);
  Perf perfAnalyzer(task);

  auto perfAttr = std::make_shared<PerfAttr>();
  perfAttr->num_running = 1000;
  perfAttr->current_timer = []() -> double {
    return static_cast<double>(std::chrono::steady_clock::now().time_since_epoch().count()) * 1e-9;
  };

  auto perfResults = std::make_shared<PerfResults>();

  perfAnalyzer.task_run(perfAttr, perfResults);
  Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    ASSERT_EQ(output_data[0], 12);
  }
}