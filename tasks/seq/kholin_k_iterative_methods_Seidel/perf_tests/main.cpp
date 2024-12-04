#include <gtest/gtest.h>

#include <seq/kholin_k_iterative_methods_Seidel/src/ops_seq.cpp>

#include "core/perf/include/perf.hpp"
#include "seq/kholin_k_iterative_methods_Seidel/include/ops_seq.hpp"
//
TEST(kholin_k_iterative_methods_Seidel_seq, test_pipeline_run) {
  const size_t count_rows = 1000;
  const size_t count_colls = 1000;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  auto testTaskSequential = std::make_shared<kholin_k_iterative_methods_Seidel_seq::TestTaskSequential>(taskDataSeq);

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
}

TEST(kholin_k_iterative_methods_Seidel_seq, test_task_run) {
  const size_t count_rows = 1000;
  const size_t count_colls = 1000;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  // Create Task
  auto testTaskSequential = std::make_shared<kholin_k_iterative_methods_Seidel_seq::TestTaskSequential>(taskDataSeq);

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
}