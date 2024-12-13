// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korneeva_e_num_of_orderly_violations/include/ops_seq.hpp"

TEST(korneeva_e_num_of_orderly_violations_seq, test_pipeline_run) {
  const int numElements = 100000000;  // Size of input data

  // Generate input and prepare output container
  std::vector<int> inputData(numElements);
  std::vector<int> outputData(1, 0);

  std::random_device randomDevice;
  std::mt19937 generator(randomDevice());
  std::uniform_int_distribution<int> dist(0, numElements);

  std::generate(inputData.begin(), inputData.end(), [&]() { return dist(generator); });

  // Configure TaskData
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Instantiate OrderlyViolationsCounter Task
  auto violationCounterTask =
      std::make_shared<korneeva_e_num_of_orderly_violations_seq::OrderlyViolationsCounter<int, int>>(taskData);

  // Set up performance attributes and timer
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto startTime = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&]() {
    auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
    return std::chrono::duration<double>(elapsed).count();
  };

  // Initialize performance results container
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Run performance analysis
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(violationCounterTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Validate result
  int expectedViolations = violationCounterTask->count_orderly_violations(inputData);
  ASSERT_EQ(expectedViolations, outputData[0]);
}

TEST(korneeva_e_num_of_orderly_violations_seq, test_task_run) {
  const int dataSize = 100000000;

  // Initialize input data with random integers and prepare output container
  std::vector<int> inputData(dataSize);
  std::vector<int> outputData(1, 0);

  std::random_device randomDevice;
  std::mt19937 engine(randomDevice());
  std::uniform_int_distribution<int> distribution(0, dataSize);

  std::generate(inputData.begin(), inputData.end(), [&]() { return distribution(engine); });

  // Configure TaskData
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Initialize OrderlyViolationsCounter Task
  auto violationCounter =
      std::make_shared<korneeva_e_num_of_orderly_violations_seq::OrderlyViolationsCounter<int, int>>(taskData);

  // Set up performance attributes with a custom timer
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto startTime = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&]() {
    auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
    return std::chrono::duration<double>(elapsed).count();
  };

  // Initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Run task and gather performance statistics
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(violationCounter);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Validate output
  int calculatedViolations = violationCounter->count_orderly_violations(inputData);
  ASSERT_EQ(calculatedViolations, outputData[0]);
}