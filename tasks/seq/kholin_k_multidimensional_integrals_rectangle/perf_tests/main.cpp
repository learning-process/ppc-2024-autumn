#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kholin_k_multidimensional_integrals_rectangle/include/ops_seq.hpp"

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_pipeline_run) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return std::cos(f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]) *
           (1 + f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]);
  };
  std::vector<double> in_lower_limits{-1, 2, -3};
  std::vector<double> in_upper_limits{8, 8, 2};
  double epsilon = 0.1;
  int n = 1;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential>(taskDataSeq);

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
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_task_run) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return std::cos(f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]) *
           (1 + f_values[0] * f_values[0] + f_values[1] * f_values[1] + f_values[2] * f_values[2]);
  };
  std::vector<double> in_lower_limits{-1, 2, -3};
  std::vector<double> in_upper_limits{8, 8, 2};
  double epsilon = 0.1;
  int n = 1;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  delete f_object;
}  //