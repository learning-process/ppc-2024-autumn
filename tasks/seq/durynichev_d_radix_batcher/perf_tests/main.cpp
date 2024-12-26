#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/durynichev_d_radix_batcher/include/ops_seq.hpp"

TEST(durynichev_d_radix_batcher_seq, pipeline_run) {
  int vector_size = std::pow(2, 18);
  std::vector<double> data;
  std::vector<double> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  data.resize(vector_size);
  result_data.resize(vector_size);

  double current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));

  taskDataSeq->inputs_count.emplace_back(vector_size);

  auto taskSequential = std::make_shared<durynichev_d_radix_batcher_seq::RadixBatcher>(taskDataSeq);

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

  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}

TEST(durynichev_d_radix_batcher_seq, task_run) {
  int vector_size = std::pow(2, 18);
  std::vector<double> data;
  std::vector<double> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  data.resize(vector_size);
  result_data.resize(vector_size);

  double current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));

  taskDataSeq->inputs_count.emplace_back(vector_size);

  auto taskSequential = std::make_shared<durynichev_d_radix_batcher_seq::RadixBatcher>(taskDataSeq);

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

  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}
