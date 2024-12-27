#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kovalchuk_a_odd_even_megre_sort/include/ops_seq.hpp"

TEST(sort_perf_test, test_pipeline_run) {
  std::vector<int> global_vec;
  std::vector<int> global_sorted_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 10000;
  global_vec = std::vector<int>(count_size_vector);
  std::generate(global_vec.begin(), global_vec.end(), std::rand);
  global_sorted_vec = std::vector<int>(count_size_vector);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sorted_vec.data()));
  taskDataSeq->outputs_count.emplace_back(global_sorted_vec.size());

  auto testTaskSequential = std::make_shared<kovalchuk_a_odd_even_seq::TestTaskSequential>(taskDataSeq);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto current_timer = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - current_timer).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(std::is_sorted(global_sorted_vec.begin(), global_sorted_vec.end()));
}

TEST(sort_perf_test, test_task_run) {
  std::vector<int> global_vec;
  std::vector<int> global_sorted_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 10000;
  global_vec = std::vector<int>(count_size_vector);
  std::generate(global_vec.begin(), global_vec.end(), std::rand);
  global_sorted_vec = std::vector<int>(count_size_vector);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sorted_vec.data()));
  taskDataSeq->outputs_count.emplace_back(global_sorted_vec.size());

  auto testTaskSequential = std::make_shared<kovalchuk_a_odd_even_seq::TestTaskSequential>(taskDataSeq);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto current_timer = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - current_timer).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(std::is_sorted(global_sorted_vec.begin(), global_sorted_vec.end()));
}
