#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/belov_a_radix_sort_with_batcher_mergesort/include/ops_seq.hpp"

using namespace belov_a_radix_batcher_mergesort_seq;

namespace belov_a_radix_batcher_mergesort_seq {
vector<bigint> generate_mixed_values_array(int n) {
  random_device rd;
  mt19937 gen(rd());

  uniform_int_distribution<bigint> small_range(-999LL, 999LL);
  uniform_int_distribution<bigint> medium_range(-10000LL, 10000LL);
  uniform_int_distribution<bigint> large_range(-10000000000LL, 10000000000LL);
  uniform_int_distribution<int> choice(0, 2);

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    if (choice(gen) == 0) {
      arr.push_back(small_range(gen));
    } else if (choice(gen) == 1) {
      arr.push_back(medium_range(gen));
    } else {
      arr.push_back(large_range(gen));
    }
  }
  return arr;
}

vector<bigint> generate_int_array(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(numeric_limits<int>::min(), numeric_limits<int>::max());

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}

vector<bigint> generate_bigint_array(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<bigint> dist(numeric_limits<bigint>::min() / 2, numeric_limits<bigint>::max() / 2);

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}
}  // namespace belov_a_radix_batcher_mergesort_seq

TEST(belov_a_radix_batcher_mergesort_perf_test, test_pipeline_run) {
  // Create data
  int n = 1048576;
  vector<bigint> arr = generate_mixed_values_array(n);

  // Create TaskData
  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<belov_a_radix_batcher_mergesort_seq::RadixBatcherMergesortSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(belov_a_radix_batcher_mergesort_perf_test, test_task_run) {
  // Create data
  int n = 1048576;
  vector<bigint> arr = generate_mixed_values_array(n);

  // Create TaskData
  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<belov_a_radix_batcher_mergesort_seq::RadixBatcherMergesortSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ppc::core::Perf::print_perf_statistic(perfResults);
}