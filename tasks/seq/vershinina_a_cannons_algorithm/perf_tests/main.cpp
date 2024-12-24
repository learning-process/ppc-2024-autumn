#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vershinina_a_cannons_algorithm/include/ops_seq.hpp"

TEST(vershinina_a_cannons_algorithm, test_pipeline_run) {
  auto lhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto rhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  auto res_c = vershinina_a_cannons_algorithm::TMatrix<double>::create(3);

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3);

  auto ref_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {14, 32, 50, 32, 77, 122, 50, 122, 194});
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(res_c.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(res.data.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data.data()));
  taskDataSeq->outputs_count.emplace_back(res.n);

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
  ASSERT_EQ(act_res, ref_res);
}

TEST(vershinina_a_cannons_algorithm, test_task_run) {
  auto lhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto rhs = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  auto res_c = vershinina_a_cannons_algorithm::TMatrix<double>::create(3);

  auto act_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3);

  auto ref_res = vershinina_a_cannons_algorithm::TMatrix<double>::create(3, {14, 32, 50, 32, 77, 122, 50, 122, 194});
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(res_c.data.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(res.data.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data.data()));
  taskDataSeq->outputs_count.emplace_back(res.n);

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
  ASSERT_EQ(act_res, ref_res);
}
