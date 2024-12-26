#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/Sdobnov_V_mergesort_Betcher/include/ops_seq.hpp"

TEST(Sdobnov_V_mergesort_Betcher_seq, test_pipeline_run) {
  long unsigned int size = 4096;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  auto test = std::make_shared<Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq>(taskDataSeq);
  ASSERT_EQ(test->validation(), true);
  test->pre_processing();
  test->run();
  test->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res.size(), size);
}

TEST(Sdobnov_V_mergesort_Betcher_seq, test_task_run) {
  long unsigned int size = 4096;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  auto test = std::make_shared<Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq>(taskDataSeq);
  ASSERT_EQ(test->validation(), true);
  test->pre_processing();
  test->run();
  test->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res.size(), size);
}