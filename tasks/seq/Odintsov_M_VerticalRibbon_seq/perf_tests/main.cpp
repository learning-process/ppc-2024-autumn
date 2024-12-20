
#include <gtest/gtest.h>

#include <seq/Odintsov_M_VerticalRibbon_seq/include/ops_seq.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"

TEST(Odintsov_M_VerticalRibbon_perf_test_seq, matrix_test_pipeline_run) {
  std::vector<double> matrixA(1000000, 1);
  std::vector<double> vectorB(1000, 1);

  std::vector<double> vectorC(1000, 1000);
  std::vector<double> out(vectorC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
  // [0] - szA [1] - rowA [2] - szB
  taskDataSeq->inputs_count.emplace_back(1000000);
  taskDataSeq->inputs_count.emplace_back(1000);
  taskDataSeq->inputs_count.emplace_back(1000);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testClass = std::make_shared<Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClass);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_EQ(vectorC, out);
}

TEST(Odintsov_M_VerticalRibbon_perf_test_seq, test_task_run) {
  std::vector<double> matrixA(1000000, 1);
  std::vector<double> vectorB(1000, 1);

  std::vector<double> vectorC(1000, 1000);
  std::vector<double> out(vectorC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
  // [0] - szA [1] - rowA [2] - szB
  taskDataSeq->inputs_count.emplace_back(1000000);
  taskDataSeq->inputs_count.emplace_back(1000);
  taskDataSeq->inputs_count.emplace_back(1000);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testClass = std::make_shared<Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClass);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_EQ(vectorC, out);
}