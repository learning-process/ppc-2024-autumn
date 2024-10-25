#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "core/perf/include/perf.hpp"
#include "seq/burykin_m_word_count/include/ops_seq.hpp"

using namespace ppc::core;
using namespace burykin_m_word_count;

TEST(WordCountSequential, TestSingleWord) {
  std::string input = "Hello.";
  int expected_count = 1;

  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  taskData->inputs_count.emplace_back(input_data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(output_data.size());

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output_data[0], expected_count);
}

TEST(WordCountSequential, TestMultipleWords) {
  std::string input = "Hello world baba gaga.";
  int expected_count = 4;
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  taskData->inputs_count.emplace_back(input_data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(output_data.size());

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output_data[0], expected_count);
}

TEST(WordCountSequential, TestApostrophes) {
  std::string input = "Feels like i'm walking on sunshine.";
  int expected_count = 6;
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  taskData->inputs_count.emplace_back(input_data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(output_data.size());

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output_data[0], expected_count);
}

TEST(WordCountSequential, TestNoWords) {
  std::string input = "!!! ??? ...";
  int expected_count = 0;
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  taskData->inputs_count.emplace_back(input_data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(output_data.size());

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output_data[0], expected_count);
}

TEST(WordCountSequential, PipelineRunPerformance) {
  std::string input = "This is a sample text to test the word counting functionality.";
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  taskData->inputs_count.emplace_back(input_data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(output_data.size());

  auto task = std::make_shared<TestTaskSequential>(taskData);
  Perf perfAnalyzer(task);

  auto perfAttr = std::make_shared<PerfAttr>();
  perfAttr->num_running = 1000;
  perfAttr->current_timer = []() -> double {
    return static_cast<double>(std::chrono::steady_clock::now().time_since_epoch().count()) * 1e-9;
  };

  auto perfResults = std::make_shared<PerfResults>();

  perfAnalyzer.pipeline_run(perfAttr, perfResults);
  Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(output_data[0], 11);
}

TEST(WordCountSequential, TaskRunPerformance) {
  std::string input = "Another example sentence to evaluate the performance of the word counting task.";
  std::vector<char> input_data(input.begin(), input.end());
  std::vector<int> output_data(1, 0);

  auto taskData = std::make_shared<TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  taskData->inputs_count.emplace_back(input_data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(output_data.size());

  auto task = std::make_shared<TestTaskSequential>(taskData);
  Perf perfAnalyzer(task);

  auto perfAttr = std::make_shared<PerfAttr>();
  perfAttr->num_running = 1000;
  perfAttr->current_timer = []() -> double {
    return static_cast<double>(std::chrono::steady_clock::now().time_since_epoch().count()) * 1e-9;
  };

  auto perfResults = std::make_shared<PerfResults>();

  perfAnalyzer.task_run(perfAttr, perfResults);
  Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(output_data[0], 12);
}
