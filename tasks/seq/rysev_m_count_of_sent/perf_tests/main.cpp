// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <string>


#include "core/perf/include/perf.hpp"
#include "seq/rysev_m_count_of_sent/include/ops_seq.hpp"

TEST(rysev_m_count_of_sent_seq, test_pipeline_run) {
  //create data
  std::string str("Have I told you what madness is?");
  std::vector<int> out(1, 0);
  for (int i = 0; i < 100; i++) str.append("Have I told you what madness is?");

  //create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testSeqTask = std::make_shared<rysev_m_count_of_sent_seq::SentCountSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(101, out[0]);
}

TEST(rysev_m_count_of_sent_seq, test_task_run) {
  //create data
  std::string str("Have I told you what madness is?");
  std::vector<int> out(1, 0);
  for (int i = 0; i < 100; i++) str.append("Have I told you what madness is?");

  //create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testSeqTask = std::make_shared<rysev_m_count_of_sent_seq::SentCountSequential>(taskDataSeq);

  // Create Perf attributesû
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);

  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(101, out[0]);
}
