#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

TEST(deryabin_m_cannons_algorithm_seq, test_pipeline_run) {
  std::vector<double> input_matrix_A = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_B = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_C = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_A[dimension * 101] = 1;
    input_matrix_B[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  auto cannons_algorithm_TaskSequential =
      std::make_shared<deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(cannons_algorithm_TaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_matrix_A[0], out_matrix_C[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_task_run) {
  std::vector<double> input_matrix_A = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_B = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_C = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_A[dimension * 101] = 1;
    input_matrix_B[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  auto cannons_algorithm_TaskSequential =
      std::make_shared<deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(cannons_algorithm_TaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_matrix_A[0], out_matrix_C[0]);
}
