// Copyright 2023 Nasedkin Egor
#include <gtest/gtest.h>

#include <boost/timer/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(seq_example_perf_test, test_pipeline_run) {
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows = 4;
  int cols = 3;
  global_matrix = std::vector<int>(rows * cols, 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());

  auto testSeqTaskSequential = std::make_shared<nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential>(taskDataSeq);
  ASSERT_EQ(testSeqTaskSequential->validation(), true);
  testSeqTaskSequential->pre_processing();
  testSeqTaskSequential->run();
  testSeqTaskSequential->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::timer::cpu_timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(global_max, std::vector<int>(3, 1));
}

TEST(seq_example_perf_test, test_task_run) {
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows = 4;
  int cols = 3;
  global_matrix = std::vector<int>(rows * cols, 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());

  auto testSeqTaskSequential = std::make_shared<nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential>(taskDataSeq);
  ASSERT_EQ(testSeqTaskSequential->validation(), true);
  testSeqTaskSequential->pre_processing();
  testSeqTaskSequential->run();
  testSeqTaskSequential->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::timer::cpu_timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(global_max, std::vector<int>(3, 1));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}