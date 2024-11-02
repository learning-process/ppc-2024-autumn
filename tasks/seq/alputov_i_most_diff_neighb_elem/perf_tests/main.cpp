// Copyright 2024 Alputov Ivan
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/alputov_i_most_diff_neighb_elem/include/ops_seq.hpp"

TEST(alputov_i_most_diff_neighb_elem_perf_test, test_pipeline_run) {
  std::vector<int> inputVector;
  int outputPair[2];
  const int count = 125000000;
  inputVector = std::vector<int>(count);
  for (size_t i = 0; i < inputVector.size(); i++) {
    inputVector[i] = i;
  }
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  auto testTaskSequential = std::make_shared<alputov_i_most_diff_neighb_elem_seq::SequentialTask>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  int expectedMaxDifferenceindex = testTaskSequential->Max_Neighbour_Seq_Pos(inputVector);
  ASSERT_EQ(std::abs(outputPair[1] - outputPair[0]),
            std::abs(inputVector[expectedMaxDifferenceindex + 1] - inputVector[expectedMaxDifferenceindex]));
}

TEST(alputov_i_most_diff_neighb_elem_perf_test, test_task_run) {
  std::vector<int> inputVector;
  int outputPair[2];
  const int count = 125000000;
  inputVector = std::vector<int>(count);
  for (size_t i = 0; i < inputVector.size(); i++) {
    inputVector[i] = i;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  auto testTaskSequential = std::make_shared<alputov_i_most_diff_neighb_elem_seq::SequentialTask>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  int expectedMaxDifferenceindex = testTaskSequential->Max_Neighbour_Seq_Pos(inputVector);
  ASSERT_EQ(std::abs(outputPair[1] - outputPair[0]),
            std::abs(inputVector[expectedMaxDifferenceindex + 1] - inputVector[expectedMaxDifferenceindex]));
}
