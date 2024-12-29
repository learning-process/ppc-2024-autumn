#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vershinina_a_cannons_algorithm/include/ops_seq.hpp"

vershinina_a_cannons_algorithm::TMatrix<double> getRandomMatrix(double r) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  auto matrix = vershinina_a_cannons_algorithm::TMatrix<double>::create(r);
  for (size_t i = 0; i < matrix.n * matrix.n; i++) {
    matrix.data[i] = distr(gen);
  }
  return matrix;
}

TEST(vershinina_a_cannons_algorithm, test_pipeline_run) {
  auto lhs = getRandomMatrix(24);
  auto rhs = getRandomMatrix(24);

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(24);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(24);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(act_res.data.data()));
  taskDataSeq->outputs_count.emplace_back(act_res.n);

  auto testTaskSequential = std::make_shared<vershinina_a_cannons_algorithm::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
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

TEST(vershinina_a_cannons_algorithm, test_task_run) {
  auto lhs = getRandomMatrix(24);
  auto rhs = getRandomMatrix(24);

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(24);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs_count.emplace_back(24);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(act_res.data.data()));
  taskDataSeq->outputs_count.emplace_back(act_res.n);

  auto testTaskSequential = std::make_shared<vershinina_a_cannons_algorithm::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
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
