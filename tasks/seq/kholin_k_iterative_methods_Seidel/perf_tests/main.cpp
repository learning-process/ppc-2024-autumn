#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/kholin_k_iterative_methods_Seidel/include/ops_seq.hpp"
//
TEST(kholin_k_iterative_methods_Seidel_seq, test_pipeline_run) {
  const size_t count_rows = 5000;
  const size_t count_colls = 5000;
  float epsilon = 0.001f;
  kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);

  float *in = new float[count_rows * count_colls];
  kholin_k_iterative_methods_Seidel_seq::copyA_(in, count_rows, count_colls);
  float *out = new float[count_rows];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
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
  const size_t count_rows = 10000;
  const size_t count_colls = 10000;
  float epsilon = 0.001f;
  kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);

  float *in = new float[count_rows * count_colls];
  kholin_k_iterative_methods_Seidel_seq::copyA_(in, count_rows, count_colls);
  float *out = new float[count_rows];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
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

// int main(int argc, char **argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
