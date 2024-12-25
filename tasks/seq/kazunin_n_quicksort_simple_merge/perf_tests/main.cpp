#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kazunin_n_quicksort_simple_merge/include/ops_seq.hpp"

TEST(kazunin_n_quicksort_simple_merge_seq, pipeline_run) {
  int vector_size = 10000;
  std::vector<int> data(vector_size);
  int current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataPar->inputs_count.emplace_back(data.size());

  result_data.resize(vector_size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataPar->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<kazunin_n_quicksort_simple_merge_seq::QuicksortSimpleMergeSeq>(taskDataPar);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
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

  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}

TEST(kazunin_n_quicksort_simple_merge_seq, task_run) {
  int vector_size = 10000;
  std::vector<int> data(vector_size);
  int current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataPar->inputs_count.emplace_back(data.size());

  result_data.resize(vector_size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataPar->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<kazunin_n_quicksort_simple_merge_seq::QuicksortSimpleMergeSeq>(taskDataPar);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
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

  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}
