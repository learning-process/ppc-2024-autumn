#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kholin_k_vector_neighbor_diff_elems/include/ops_seq.hpp"

TEST(kholin_k_vector_neighbor_diff_elems_seq, test_pipeline_run) {
  // Create data
  const int count = 20000000;

  std::vector<int32_t> in(count, 1);      // in data
  std::vector<int32_t> out(2, 0);         // out data
  std::vector<uint64_t> out_index(2, 0);  // out data
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<kholin_k_vector_neighbor_diff_elems_seq::MostDiffNeighborElements<int32_t, uint64_t>>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;                                 // num launches programm
  const auto t0 = std::chrono::high_resolution_clock::now();  // set timer now
  perfAttr->current_timer = [&] {
    auto current_time_point =
        std::chrono::high_resolution_clock::now();  // use timer  chrono and calculate difference between t0 and now
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();  // get
                                                                                                            // result
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();  // results perfomance

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(kholin_k_vector_neighbor_diff_elems_seq, test_task_run) {
  // Create data
  const int count = 250000000;

  std::vector<int32_t> in(count, 1);      // in data
  std::vector<int32_t> out(2, 0);         // out data
  std::vector<uint64_t> out_index(2, 0);  // out data
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<kholin_k_vector_neighbor_diff_elems_seq::MostDiffNeighborElements<int32_t, uint64_t>>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;                                 // num launches programm
  const auto t0 = std::chrono::high_resolution_clock::now();  // set timer now
  perfAttr->current_timer = [&] {
    auto current_time_point =
        std::chrono::high_resolution_clock::now();  // use timer  chrono and calculate difference between t0 and now
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();  // get
                                                                                                            // result
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();  // results perfomance

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
