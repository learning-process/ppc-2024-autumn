#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/dormidontov_e_min_value_by_columns_seq/include/ops_seq.hpp"

TEST(dormidontov_e_min_value_by_columns_seq, test_pipeline_run) {
  int rs = 3000;
  int cs = 3000;
  std::vector<int> matrix(rs * cs);
  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < cs; ++j) {
      matrix[i * cs + j] = i * 1000 + j;
    }
  }

  std::vector<int> res_out(cs, 0);
  std::vector<int> exp_res(cs);
  for (int j = 0; j < cs; ++j) {
    exp_res[j] = j;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // Create Task
  auto testTaskSequential = std::make_shared<dormidontov_e_min_value_by_columns_seq::TestTaskSequential>(taskDataSeq);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
  taskDataSeq->outputs_count.emplace_back(res_out.size());

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

  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_min_value_by_columns_seq, test_task_run) {
  int rs = 3000;
  int cs = 3000;
  std::vector<int> matrix(rs * cs);
  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < cs; ++j) {
      matrix[i * cs + j] = i * 1000 + j;
    }
  }
  std::vector<int> res_out(cs, 0);
  std::vector<int> exp_res(cs);
  for (int j = 0; j < cs; ++j) {
    exp_res[j] = j;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<dormidontov_e_min_value_by_columns_seq::TestTaskSequential>(taskDataSeq);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
  taskDataSeq->outputs_count.emplace_back(res_out.size());

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
  ASSERT_EQ(res_out, exp_res);
}