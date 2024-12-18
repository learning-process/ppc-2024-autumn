#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "mpi/kapustin_i_bubble/include/avg_mpi.hpp"
TEST(kapustin_i_bubble_sort_mpi, pipeline) {
  boost::mpi::communicator world;

  const int data_size = 45000;
  std::vector<int> input_data(data_size);
  std::vector<int> output_data(data_size);

  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000, 1000);
  for (int i = 0; i < data_size; ++i) {
    input_data[i] = distribution(generator);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataMPI->inputs_count.emplace_back(data_size);
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataMPI->outputs_count.emplace_back(data_size);
  }

  auto bubbleSortMPI = std::make_shared<kapustin_i_bubble_sort_mpi::BubbleSortMPI>(taskDataMPI);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bubbleSortMPI);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> expected = input_data;
    std::sort(expected.begin(), expected.end());

    std::memcpy(output_data.data(), taskDataMPI->outputs[0], data_size * sizeof(int));
    ASSERT_EQ(output_data, expected);
  }
}

TEST(kapustin_i_bubble_sort_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int data_size = 30000;
  std::vector<int> input_data(data_size);
  std::vector<int> output_data(data_size);

  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000000, 1000000);
  for (int i = 0; i < data_size; ++i) {
    input_data[i] = distribution(generator);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataMPI->inputs_count.emplace_back(input_data.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataMPI->outputs_count.emplace_back(output_data.size());
  }

  auto bubbleSortMPI = std::make_shared<kapustin_i_bubble_sort_mpi::BubbleSortMPI>(taskDataMPI);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bubbleSortMPI);

  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> expected = input_data;
    std::sort(expected.begin(), expected.end());

    std::memcpy(output_data.data(), taskDataMPI->outputs[0], data_size * sizeof(int));
    ASSERT_EQ(output_data, expected);
  }
}