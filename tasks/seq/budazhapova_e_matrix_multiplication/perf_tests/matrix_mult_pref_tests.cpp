#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/budazhapova_e_matrix_multiplication/include/matrix_mult.hpp"

TEST(budazhapova_e_matrix_mult_seq, test_pipeline_run) {
  std::vector<int> A_matrix(8000000, 2);
  std::vector<int> b_vector(2000, 3);
  std::vector<int> out(4000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(A_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  taskDataSeq->inputs_count.emplace_back(b_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<budazhapova_e_matrix_mult_seq::MatrixMultSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
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
}

TEST(budazhapova_e_matrix_mult_seq, test_task_run) {
  std::vector<int> A_matrix(8000000, 2);
  std::vector<int> b_vector(2000, 3);
  std::vector<int> out(4000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(A_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  taskDataSeq->inputs_count.emplace_back(b_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<budazhapova_e_matrix_mult_seq::MatrixMultSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
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
}
