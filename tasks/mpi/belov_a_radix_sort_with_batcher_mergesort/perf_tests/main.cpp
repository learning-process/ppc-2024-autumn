#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/belov_a_radix_sort_with_batcher_mergesort/include/ops_mpi.hpp"

using namespace belov_a_radix_batcher_mergesort_mpi;

namespace belov_a_radix_batcher_mergesort_mpi {
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
}  // namespace belov_a_radix_batcher_mergesort_mpi

TEST(belov_a_radix_batcher_mergesort_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int n = 1048576;
  vector<bigint> arr = generate_mixed_values_array(n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  // Create Task
  auto testMpiTaskParallel =
      std::make_shared<belov_a_radix_batcher_mergesort_mpi::RadixBatcherMergesortParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    auto testMpiTaskSequential =
        std::make_shared<belov_a_radix_batcher_mergesort_mpi::RadixBatcherMergesortSequential>(taskDataSeq);

    ASSERT_TRUE(testMpiTaskSequential->validation());
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(belov_a_radix_batcher_mergesort_perf_test, test_task_run) {
  boost::mpi::communicator world;

  int n = 1048576;
  vector<bigint> arr = generate_mixed_values_array(n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  // Create Task
  auto testMpiTaskParallel =
      std::make_shared<belov_a_radix_batcher_mergesort_mpi::RadixBatcherMergesortParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    auto testMpiTaskSequential =
        std::make_shared<belov_a_radix_batcher_mergesort_mpi::RadixBatcherMergesortSequential>(taskDataSeq);

    ASSERT_TRUE(testMpiTaskSequential->validation());
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}