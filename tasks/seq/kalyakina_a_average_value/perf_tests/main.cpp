// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/kalyakina_a_average_value/include/ops_seq.hpp"

std::vector<int> RandomVectorWithFixSum(int sum, const int &count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> result_vector(count);
  for (int i = 0; i < count - 1; i++) {
    result_vector[i] = gen() % (std::min(sum, 255) - 1);
    sum -= result_vector[i];
  }
  result_vector[count - 1] = sum;
  return result_vector;
}

TEST(kalyakina_a_average_value_seq, test_pipeline_run) {
  const int count = 100;
  const int sum = 20000;
  const double expected_value = (double)sum / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto AverageValueTaskSequential =
      std::make_shared<kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(AverageValueTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}

TEST(kalyakina_a_average_value_seq, test_task_run) {
  const int count = 100;
  const int sum = 20000;
  const double expected_value = (double)sum / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto AverageValueTaskSequential =
      std::make_shared<kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(AverageValueTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}
