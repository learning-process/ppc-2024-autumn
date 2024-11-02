// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_seq, test_pipeline_run) {
  const size_t length = 10;
  const int alpha = 1;
  // Create data
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  // Create Task
  auto testTaskSequential =
      std::make_shared<kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int>>(taskSeq);
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  size_t result = 0;
  ASSERT_EQ(result, out[0]);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
