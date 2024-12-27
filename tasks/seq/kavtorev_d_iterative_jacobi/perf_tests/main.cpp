#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kavtorev_d_iterative_jacobi/include/ops_seq.hpp"

namespace kavtorev_d_iterative_jacobi_seq {

void generateTestData(size_t n, std::vector<double>& A_flat, std::vector<double>& F, std::vector<double>& x) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> offDiagDist(-10.0, 10.0);
  std::uniform_real_distribution<double> diagDist(20.0, 50.0);
  std::uniform_real_distribution<double> solDist(-100.0, 100.0);

  x.resize(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = solDist(gen);
  }

  A_flat.resize(n * n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        double value = offDiagDist(gen);
        A_flat[i * n + j] = value;
        sum += std::abs(value);
      }
    }

    A_flat[i * n + i] = sum + diagDist(gen);
  }

  F.resize(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += A_flat[i * n + j] * x[j];
    }
    F[i] = sum;
  }
}

}  // namespace kavtorev_d_iterative_jacobi_seq

TEST(kavtorev_d_iterative_jacobi_seq, pipeline_run) {
  int n = 1000;
  double eps = 0.001;
  int iterations = 1000;

  std::vector<double> A;
  std::vector<double> F;
  std::vector<double> exp_X;
  kavtorev_d_iterative_jacobi_seq::generateTestData(n, A, F, exp_X);
  std::vector<double> seq_X(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
  taskDataSeq->inputs_count.emplace_back(F.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_X.data()));
  taskDataSeq->outputs_count.emplace_back(seq_X.size());

  auto taskSequential = std::make_shared<kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  bool seq_run_res = taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
  if (seq_run_res) {
    ASSERT_EQ(seq_X.size(), exp_X.size());
    double error = 0.0;
    for (int i = 0; i < n; ++i) {
      error = std::max(error, std::abs(exp_X[i] - seq_X[i]));
    }
    EXPECT_LT(error, eps);
  }
}

TEST(kavtorev_d_iterative_jacobi_seq, task_run) {
  int n = 1000;
  double eps = 0.001;
  int iterations = 1000;

  std::vector<double> A;
  std::vector<double> F;
  std::vector<double> exp_X;
  kavtorev_d_iterative_jacobi_seq::generateTestData(n, A, F, exp_X);
  std::vector<double> seq_X(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
  taskDataSeq->inputs_count.emplace_back(F.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_X.data()));
  taskDataSeq->outputs_count.emplace_back(seq_X.size());

  auto taskSequential = std::make_shared<kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  bool seq_run_res = taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
  if (seq_run_res) {
    ASSERT_EQ(seq_X.size(), exp_X.size());
    double error = 0.0;
    for (int i = 0; i < n; ++i) {
      error = std::max(error, std::abs(exp_X[i] - seq_X[i]));
    }
    EXPECT_LT(error, eps);
  }
}
