#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/rams_s_gaussian_elimination_horizontally/include/ops_seq.hpp"

void rams_s_gaussian_elimination_horizontally_seq_run_perf_test(bool pipeline) {
  size_t cols_count = 501;
  size_t rows_count = cols_count - 1;
  std::vector<double> in(rows_count * cols_count, 0);
  for (size_t row = 0; row < rows_count; row++) {
    double sum = 0;
    size_t col = 0;
    for (; col < cols_count - 1; col++) {
      sum += (in[row * cols_count + col] = std::min(col, row) + 1);
    }
    in[row * cols_count + col] = -sum;
  }
  std::vector<double> out(rows_count, 0);
  std::vector<double> expected(rows_count, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<rams_s_gaussian_elimination_horizontally_seq::TaskSequential>(taskDataSeq);

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
  if (pipeline) {
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  } else {
    perfAnalyzer->task_run(perfAttr, perfResults);
  }
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < expected.size(); i++) {
    if (std::isnan(expected[i])) {
      ASSERT_TRUE(std::isnan(out[i]));
    } else {
      ASSERT_DOUBLE_EQ(expected[i], out[i]);
    }
  }
}

TEST(rams_s_gaussian_elimination_horizontally_seq_perf_test, test_pipeline_run) {
  rams_s_gaussian_elimination_horizontally_seq_run_perf_test(true);
}

TEST(rams_s_gaussian_elimination_horizontally_seq_perf_test, test_task_run) {
  rams_s_gaussian_elimination_horizontally_seq_run_perf_test(false);
}
