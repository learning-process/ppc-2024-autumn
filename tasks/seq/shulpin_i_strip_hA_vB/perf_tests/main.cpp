#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shulpin_i_strip_hA_vB/include/strip_hA_vB.hpp"


TEST(shulpin_strip_scheme_A_B, pipeline_run) {
  const int rows_a = 500;
  const int cols_a = 500;
  const int rows_b = 500;
  const int cols_b = 500;

  std::vector<int> A(rows_a * cols_a, 1);
  std::vector<int> B(rows_b * cols_b, 1);
  std::vector<int> C(rows_a * cols_b, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  auto TaskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  int ref_res = static_cast<int>(rows_a) * static_cast<int>(cols_b);

  ASSERT_EQ(ref_res, C.size());
}

TEST(shulpin_strip_scheme_A_B, task_run) {
  const int rows_a = 500;
  const int cols_a = 500;
  const int rows_b = 500;
  const int cols_b = 500;

  std::vector<int> A(rows_a * cols_a, 1);
  std::vector<int> B(rows_b * cols_b, 1);
  std::vector<int> C(rows_a * cols_b, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  auto TaskSequential = std::make_shared<shulpin_strip_scheme_A_B::Matrix_hA_vB_seq>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  int ref_res = static_cast<int>(rows_a) * static_cast<int>(cols_b);

  ASSERT_EQ(ref_res, C.size());
}
