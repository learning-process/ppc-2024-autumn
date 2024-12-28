// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/ermolaev_v_multidimensional_integral_rectangle/include/ops_seq.hpp"

namespace ermolaev_v_multidimensional_integral_rectangle_seq {

void testBody(std::vector<std::pair<double, double>> limits,
              ermolaev_v_multidimensional_integral_rectangle_seq::function func,
              ppc::core::PerfResults::TypeOfRunning type, double eps = 1e-4) {
  double out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto testTaskSequential =
      std::make_shared<ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential>(taskDataSeq, func);

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
  if (type == ppc::core::PerfResults::PIPELINE)
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  else if (type == ppc::core::PerfResults::TASK_RUN)
    perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}

double testFunc(std::vector<double> &args) { return args.at(0); }
}  // namespace ermolaev_v_multidimensional_integral_rectangle_seq
namespace erm_integral_seq = ermolaev_v_multidimensional_integral_rectangle_seq;

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, test_pipeline_run) {
  std::vector<std::pair<double, double>> limits(10, {-7, 7});
  erm_integral_seq::testBody(limits, erm_integral_seq::testFunc, ppc::core::PerfResults::PIPELINE);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, test_task_run) {
  std::vector<std::pair<double, double>> limits(10, {-7, 7});
  erm_integral_seq::testBody(limits, erm_integral_seq::testFunc, ppc::core::PerfResults::TASK_RUN);
}