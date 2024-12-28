#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/tyshkevich_a_monte_carlo/include/ops_seq.hpp"
#include "seq/tyshkevich_a_monte_carlo/include/test_include.hpp"

TEST(tyshkevich_a_monte_carlo_seq, test_pipeline_run) {
  std::function<double(const std::vector<double> &)> function = tyshkevich_a_monte_carlo_seq::function_sin_sum;
  double exp_res = 1.379093;
  int dimensions = 3;
  double precision = 1000000;
  double left_bound = 0.0;
  double right_bound = 1.0;
  double result = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimensions));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&precision));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right_bound));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  auto task = std::make_shared<tyshkevich_a_monte_carlo_seq::MonteCarloSequential>(taskDataSeq, std::move(function));

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(result, exp_res, 0.1);
}

TEST(tyshkevich_a_monte_carlo_seq, test_task_run) {
  std::function<double(const std::vector<double> &)> function = tyshkevich_a_monte_carlo_seq::function_sin_sum;
  double exp_res = 1.379093;
  int dimensions = 3;
  double precision = 1000000;
  double left_bound = 0.0;
  double right_bound = 1.0;
  double result = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimensions));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&precision));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right_bound));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  auto task = std::make_shared<tyshkevich_a_monte_carlo_seq::MonteCarloSequential>(taskDataSeq, std::move(function));

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(result, exp_res, 0.1);
}
