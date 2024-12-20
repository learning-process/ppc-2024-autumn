#include <gtest/gtest.h>

#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/gromov_a_gaussian_method_vertical/include/ops_seq.hpp"

TEST(gromov_a_gaussian_method_vertical_seq, test_pipeline_run) {
  int equations = 1500;
  int band_width = 10;

  int size_coefficient_mat = equations * equations;
  std::vector<int> input_coefficient(size_coefficient_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  auto gaussVerticalSequential =
      std::make_shared<gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential>(taskDataSeq, band_width);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussVerticalSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_EQ(int(func_res.size()), equations);
}

TEST(gromov_a_gaussian_method_vertical_seq, test_pipeline_run2) {
  int equations = 1000;
  int band_width = 15;

  int size_coefficient_mat = equations * equations;
  std::vector<int> input_coefficient(size_coefficient_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
  taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
  taskDataSeq->inputs_count.emplace_back(input_rhs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  auto gaussVerticalSequential =
      std::make_shared<gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential>(taskDataSeq, band_width);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussVerticalSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_EQ(int(func_res.size()), equations);
}
