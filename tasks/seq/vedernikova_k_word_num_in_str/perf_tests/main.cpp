#include <gtest/gtest.h>

#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"

void run_test(std::string &&in, size_t solution,
              const std::function<void(ppc::core::Perf &, const std::shared_ptr<ppc::core::PerfAttr>,
                                       const std::shared_ptr<ppc::core::PerfResults>)> &executor) {
  size_t out = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<vedernikova_k_word_num_in_str_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perfAnalyzer(testTaskSequential);
  executor(perfAnalyzer, perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  EXPECT_EQ(out, solution);
}
void run_test(const std::function<void(ppc::core::Perf &, const std::shared_ptr<ppc::core::PerfAttr>,
                                       const std::shared_ptr<ppc::core::PerfResults>)> &executor) {
  run_test("Sentence for word counter test", 5, executor);
}

TEST(vedernikova_k_word_num_in_str_mpi_perf_test, test_pipeline_run) {
  run_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.pipeline_run(perfAttr, perfResults);
  });
}

TEST(vedernikova_k_word_num_in_str_mpi_perf_test, test_task_run) {
  run_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.task_run(perfAttr, perfResults);
  });
}
