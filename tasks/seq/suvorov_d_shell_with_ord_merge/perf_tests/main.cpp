// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/suvorov_d_shell_with_ord_merge/include/ops_seq.hpp"

namespace suvorov_d_shell_with_ord_merge_seq {
std::vector<int> get_rand_vector(const size_t vec_size, const int min_int = -10000, const int max_int = 10000) {
  std::random_device r_dev;
  std::mt19937 gen(r_dev());
  std::uniform_int_distribution<> distr(min_int, max_int);

  std::vector<int> new_vec(vec_size);
  std::generate(new_vec.begin(), new_vec.end(), [&]() { return distr(gen); });
  std::sort(new_vec.begin(), new_vec.end(), std::greater<>());
  return new_vec;
}
}  // namespace suvorov_d_shell_with_ord_merge_seq

TEST(suvorov_d_shell_with_ord_merge_seq, test_pipeline_run) {
  const size_t count_of_elems = 10000000;
  std::vector<int> data_to_sort = suvorov_d_shell_with_ord_merge_seq::get_rand_vector(count_of_elems);
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(sorted_result.size());

  auto ShellSortSeq = std::make_shared<suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq>(taskDataForSortingSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_TRUE(std::is_sorted(sorted_result.begin(), sorted_result.end()));
}

TEST(suvorov_d_shell_with_ord_merge_seq, test_task_run) {
  const size_t count_of_elems = 10000000;
  std::vector<int> data_to_sort = suvorov_d_shell_with_ord_merge_seq::get_rand_vector(count_of_elems);
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(sorted_result.size());

  auto ShellSortSeq = std::make_shared<suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq>(taskDataForSortingSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_TRUE(std::is_sorted(sorted_result.begin(), sorted_result.end()));
}
