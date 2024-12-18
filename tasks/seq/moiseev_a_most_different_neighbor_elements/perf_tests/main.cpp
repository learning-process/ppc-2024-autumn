#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/moiseev_a_most_different_neighbor_elements/include/ops_seq.hpp"

TEST(moiseev_a_most_different_neighbor_elements_seq_test, test_pipeline_run) {
  const int num_elements = 10000000;

  std::vector<int32_t> in(num_elements);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);

  for (size_t i = 0; i < num_elements; ++i) {
    in[i] = i * 2;
  }
  in[0] = -5000;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  auto testTask = std::make_shared<
      moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int32_t>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out[0], -5000);
  ASSERT_EQ(out[1], 2);
}

TEST(moiseev_a_most_different_neighbor_elements_seq_test, test_task_run) {
  const int num_elements = 1000000;

  std::vector<int32_t> in(num_elements);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);

  for (size_t i = 0; i < num_elements; ++i) {
    in[i] = i * 2;
  }
  in[0] = -5000;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  auto testTask = std::make_shared<
      moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int32_t>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out[0], -5000);
  ASSERT_EQ(out[1], 2);
}
