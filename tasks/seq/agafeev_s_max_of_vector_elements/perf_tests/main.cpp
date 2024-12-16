#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

template <typename T>
static std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 200 - 100;

  return matrix;
}

TEST(agafeev_s_max_of_vector_elements, test_pipeline_run) {
  const int n = 1000;
  const int m = 1000;

  // Credate Data
  std::vector<int> in_matrix = create_RandomMatrix<int>(n, m);
  std::vector<int> out(1, 99);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
  taskData->inputs_count.emplace_back(in_matrix.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequental = std::make_shared<agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int>>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequental);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  auto temp = agafeev_s_max_of_vector_elements_seq::get_MaxValue(in_matrix);

  ASSERT_EQ(temp, out[0]);
}

TEST(agafeev_s_max_of_vector_elements, test_task_run) {
  const int n = 1000;
  const int m = 1000;

  // Credate Data
  std::vector<int> in_matrix = create_RandomMatrix<int>(n, m);
  std::vector<int> out(1, 99);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
  taskData->inputs_count.emplace_back(in_matrix.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequental = std::make_shared<agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int>>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequental);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  auto temp = agafeev_s_max_of_vector_elements_seq::get_MaxValue(in_matrix);

  ASSERT_EQ(temp, out[0]);
}
