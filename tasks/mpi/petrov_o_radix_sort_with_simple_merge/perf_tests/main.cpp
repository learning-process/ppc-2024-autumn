#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_o_radix_sort_with_simple_merge/include/ops_mpi.hpp"

using namespace petrov_o_radix_sort_with_simple_merge_mpi;

TEST(petrov_o_radix_sort_with_simple_merge_mpi, test_pipeline_run_mpi) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 100000;

  if (world.rank() == 0) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    out.resize(in.size(), 0);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel = std::make_shared<TaskParallel>(taskDataMPI);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(expected, out) << "Pipeline run MPI: Sorted array does not match the expected result.";
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, test_task_run_mpi) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 100000;

  if (world.rank() == 0) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    out.resize(in.size(), 0);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel = std::make_shared<TaskParallel>(taskDataMPI);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(expected, out) << "Task run MPI: Sorted array does not match the expected result.";
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, test_pipeline_run_seq) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    std::vector<int> in;
    std::vector<int> out;
    const size_t array_size = 100000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    out.resize(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    auto testTaskSequential = std::make_shared<TaskSequential>(taskDataSeq);

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

    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out) << "Pipeline run sequential: Sorted array does not match the expected result.";
  }
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, test_task_run_seq) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    std::vector<int> in;
    std::vector<int> out;
    const size_t array_size = 100000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    out.resize(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    auto testTaskSequential = std::make_shared<TaskSequential>(taskDataSeq);

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

    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out) << "Task run sequential: Sorted array does not match the expected result.";
  }
}