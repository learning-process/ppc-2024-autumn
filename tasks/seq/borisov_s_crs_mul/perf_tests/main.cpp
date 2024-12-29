#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/borisov_s_crs_mul/include/ops_seq.hpp"

static void generate_dense_matrix(int M, int N, double density, std::vector<double>& dense) {
  std::mt19937_64 gen(42);
  std::uniform_real_distribution<double> dist_val(0.1, 10.0);
  std::uniform_real_distribution<double> dist_density(0.0, 1.0);

  dense.resize(M * N, 0.0);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (dist_density(gen) < density) {
        dense[(i * N) + j] = dist_val(gen);
      }
    }
  }
}

static void dense_to_crs(const std::vector<double>& dense, int M, int N, std::vector<double>& values,
                         std::vector<int>& col_index, std::vector<int>& row_ptr) {
  row_ptr.resize(M + 1, 0);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double val = dense[(i * N) + j];
      if (val != 0.0) {
        values.push_back(val);
        col_index.push_back(j);
      }
    }
    row_ptr[i + 1] = static_cast<int>(values.size());
  }
}

TEST(borisov_s_crs_seq_test, Test_Pipeline_Run) {
  const int M = 3000;
  const int N = 3000;
  const int K = 3000;
  const double density = 0.01;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;
  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};

  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};

  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  auto seqTask = std::make_shared<borisov_s_crs_mul::CrsMatrixMulTask>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seqTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(borisov_s_crs_seq_test, Test_Task_Run) {
  const int M = 3000;
  const int N = 3000;
  const int K = 3000;
  const double density = 0.01;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;
  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};

  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};

  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  auto seqTask = std::make_shared<borisov_s_crs_mul::CrsMatrixMulTask>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seqTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
