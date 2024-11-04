#include <gtest/gtest.h>
#include <vector>
#include <boost/mpi/timer.hpp>
#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(seq_matrix_column_max_value_perf_test, test_pipeline_run) {
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  const int rows = 120;
  const int cols = 3;
  global_matrix = std::vector<int>(rows * cols, 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());

  auto matrixColumnMaxSeq = std::make_shared<nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSeq>(taskDataSeq);
  ASSERT_EQ(matrixColumnMaxSeq->validation(), true);
  matrixColumnMaxSeq->pre_processing();
  matrixColumnMaxSeq->run();
  matrixColumnMaxSeq->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&current_timer] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixColumnMaxSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::vector<int> reference_max(3, 1);
  ASSERT_EQ(reference_max, global_max);
}

TEST(seq_matrix_column_max_value_perf_test, test_task_run) {
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  const int rows = 120;
  const int cols = 3;
  global_matrix = std::vector<int>(rows * cols, 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());

  auto matrixColumnMaxSeq = std::make_shared<nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSeq>(taskDataSeq);
  ASSERT_EQ(matrixColumnMaxSeq->validation(), true);
  matrixColumnMaxSeq->pre_processing();
  matrixColumnMaxSeq->run();
  matrixColumnMaxSeq->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&current_timer] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixColumnMaxSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::vector<int> reference_max(3, 1);
  ASSERT_EQ(reference_max, global_max);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}