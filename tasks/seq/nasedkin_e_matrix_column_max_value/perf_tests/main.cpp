#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(nasedkin_e_matrix_column_max_value_seq, test_pipeline_run) {
  int numCols = 2000;
  int numRows = 5000;

  // Create data
  std::vector<int> matrix(numRows * numCols, 1);
  std::vector<int> result(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSequential->outputs_count.emplace_back(result.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<nasedkin_e_matrix_column_max_value_seq::TestTaskSequential>(taskDataSequential);

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
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(1, result[0]);
  }
}

TEST(nasedkin_e_matrix_column_max_value_seq, test_task_run) {
  int numRows;
  int numCols;

  // Create data
  numRows = 5000;
  numCols = 2000;
  std::vector<int> matrix(numRows * numCols, 1);
  std::vector<int32_t> res(numCols, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSequential->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<nasedkin_e_matrix_column_max_value_seq::TestTaskSequential>(taskDataSequential);

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
  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_EQ(1, res[0]);
  }
}