#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "seq/Sdobnov_V_sum_of_vector_elements/include/ops_seq.hpp"

TEST(Sdobnov_V_sum_of_vector_elements_seq, test_pipeline_run) {
  int rows = 10000;
  int columns = 10000;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  int sum = 0;
  for (std::vector<int> vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // Create Task
  auto test = std::make_shared<Sdobnov_V_sum_of_vector_elements::SumVecElemSequential>(taskDataPar);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res, sum);
}
TEST(Sdobnov_V_sum_of_vector_elements_seq, test_task_run) {
  int rows = 10000;
  int columns = 10000;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  int sum = 0;
  for (std::vector<int> vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // Create Task
  auto test = std::make_shared<Sdobnov_V_sum_of_vector_elements::SumVecElemSequential>(taskDataPar);
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(sum, res);
}