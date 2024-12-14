#include <chrono>
#include <thread>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/khasanyanov_k_average_vector/include/avg_seq.hpp"

//=========================================sequence=========================================

const int SIZE = 1220000;

TEST(khasanyanov_k_average_vector_seq, test_pipeline_run) {
  std::vector<int> global_vec(SIZE, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_seq::create_task_data<int, double>(global_vec, average);

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<int, double>>(taskData);

  RUN_TASK(*testAvgVectorSequence);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(4, average[0]);
}

TEST(khasanyanov_k_average_vector_seq, test_task_run) {
  std::vector<int> global_vec(SIZE, 4);
  std::vector<double> average(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_seq::create_task_data<int, double>(global_vec, average);

  auto testAvgVectorSequence =
      std::make_shared<khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<int, double>>(taskData);

  RUN_TASK(*testAvgVectorSequence);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testAvgVectorSequence);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(4, average[0]);
}