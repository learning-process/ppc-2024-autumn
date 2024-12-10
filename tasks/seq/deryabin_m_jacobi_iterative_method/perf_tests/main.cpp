#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/deryabin_m_jacobi_iterative_method/include/ops_seq.hpp"

TEST(deryabin_m_jacobi_iterative_method_seq, test_pipeline_run) {
  std::vector<double> input_matrix_ = std::vector<double>(10000);
  std::vector<double> input_right_vector_ = std::vector<double>(100);
  std::vector<double> output_x_vector_ = std::vector<double>(100, 0);
  for (unsigned short razmernost = 0; razmernost < 10000; razmernost++) {
    if (razmernost < 100) {
      input_right_vector_[razmernost] = razmernost + 1;
    }
    if (razmernost % 101 == 0) {
      input_matrix_[razmernost] = 1;
    } else {
      input_matrix_[razmernost] = 0;
    }
  }
  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  auto jacobi_iterative_method_TaskSequential =
      std::make_shared<deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobi_iterative_method_TaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_right_part[0], out_x_vec[0]);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_task_run) {
  std::vector<double> input_matrix_ = std::vector<double>(10000);
  std::vector<double> input_right_vector_ = std::vector<double>(100);
  std::vector<double> output_x_vector_ = std::vector<double>(100, 0);
  for (unsigned short razmernost = 0; razmernost < 10000; razmernost++) {
    if (razmernost < 100) {
      input_right_vector_[razmernost] = razmernost + 1;
    }
    if (razmernost % 101 == 0) {
      input_matrix_[razmernost] = 1;
    } else {
      input_matrix_[razmernost] = 0;
    }
  }
  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  auto jacobi_iterative_method_TaskSequential =
      std::make_shared<deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobi_iterative_method_TaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_right_part[0], out_x_vec[0]);
}
