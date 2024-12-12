#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kazunin_n_quicksort_simple_merge/include/ops_seq.hpp"

namespace kazunin_n_quicksort_simple_merge_seq {

std::vector<int> generate_random_vector(int n, int min_val = -100, int max_val = 100,
                                        unsigned seed = std::random_device{}()) {
  static std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(min_val, max_val);

  std::vector<int> vec(n);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
  return vec;
}

}  // namespace kazunin_n_quicksort_simple_merge_seq

TEST(kazunin_n_quicksort_simple_merge_seq, pipeline_run) {
  int vector_size = 100000;
  std::vector<int> data = kazunin_n_quicksort_simple_merge_seq::generate_random_vector(vector_size);
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataPar->inputs_count.emplace_back(data.size());

  result_data.resize(vector_size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataPar->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<kazunin_n_quicksort_simple_merge_seq::QuicksortSimpleMergeSeq>(taskDataPar);

  if (taskSequential->validation()) {
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
}

TEST(kazunin_n_quicksort_simple_merge_seq, task_run) {
  int vector_size = 100000;
  std::vector<int> data = kazunin_n_quicksort_simple_merge_seq::generate_random_vector(vector_size);
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataPar->inputs_count.emplace_back(data.size());

  result_data.resize(vector_size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataPar->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<kazunin_n_quicksort_simple_merge_seq::QuicksortSimpleMergeSeq>(taskDataPar);

  if (taskSequential->validation()) {
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
}
