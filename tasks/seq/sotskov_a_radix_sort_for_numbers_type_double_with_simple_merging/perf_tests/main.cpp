#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging/include/ops_seq.hpp"

std::vector<double> generate_random_input_with_same_integer_part(size_t size, int integer_part = 1,
                                                                 double min_fraction = 0.0, double max_fraction = 1.0) {
  std::vector<double> input(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_fraction, max_fraction);

  for (size_t i = 0; i < size; ++i) {
    input[i] = integer_part + dis(gen);
  }
  return input;
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, test_pipeline_run) {
  const int count = 500000;
  std::vector<double> in = generate_random_input_with_same_integer_part(count, 1, 0.0, 0.99);
  std::vector<double> out(count);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential>(
          taskDataSeq);

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
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<double> sorted_in = in;
  std::sort(sorted_in.begin(), sorted_in.end());
  ASSERT_EQ(out, sorted_in);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, test_task_run) {
  const int count = 500000;
  std::vector<double> in = generate_random_input_with_same_integer_part(count, 1, 0.0, 0.99);
  std::vector<double> out(count);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential>(
          taskDataSeq);

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

  std::vector<double> sorted_in = in;
  std::sort(sorted_in.begin(), sorted_in.end());
  ASSERT_EQ(out, sorted_in);
}
