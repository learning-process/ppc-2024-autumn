#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korablev_v_jacobi_method/include/ops_seq.hpp"

TEST(korablev_v_jacobi_method, test_pipeline_run) {
  const size_t matrix_size = 1000;

  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data(matrix_size * matrix_size, 1.0);
  std::vector<double> vector_data(matrix_size, 1.0);
  std::vector<double> out(matrix_size, 0.0);

  for (size_t i = 0; i < matrix_size; ++i) {
    matrix_data[i * matrix_size + i] = static_cast<double>(matrix_size);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto jacobiTaskSequential = std::make_shared<korablev_v_jacobi_method_seq::JacobiMethodSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobiTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(matrix_size, out.size());
}

TEST(korablev_v_jacobi_method, test_task_run) {
  const size_t matrix_size = 1000;

  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data(matrix_size * matrix_size, 1.0);
  std::vector<double> vector_data(matrix_size, 1.0);
  std::vector<double> out(matrix_size, 0.0);

  for (size_t i = 0; i < matrix_size; ++i) {
    matrix_data[i * matrix_size + i] = static_cast<double>(matrix_size);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto jacobiTaskSequential = std::make_shared<korablev_v_jacobi_method_seq::JacobiMethodSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobiTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(matrix_size, out.size());
}