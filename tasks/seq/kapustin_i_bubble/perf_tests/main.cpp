#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/kapustin_i_bubble/include/avg_seq.hpp"

TEST(kapustin_i_bubble_sort_seq, pipeline) {
  const int data_size = 40000;

  std::vector<int> input_data(data_size);
  std::vector<int> output_data(data_size);
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000000, 1000000);

  for (int i = 0; i < data_size; ++i) {
    input_data[i] = distribution(generator);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(data_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(data_size);

  auto bubbleSort = std::make_shared<kapustin_i_bubble_sort_seq::BubbleSortSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bubbleSort);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(input_data.begin(), input_data.end());
  std::memcpy(output_data.data(), taskDataSeq->outputs[0], data_size * sizeof(int));
  ASSERT_EQ(output_data, input_data);
}

TEST(kapustin_i_bubble_sort_seq, test_task_run) {
  const int data_size = 20000;

  std::vector<int> input_data(data_size);
  std::vector<int> output_data(data_size);
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000000, 1000000);

  for (int i = 0; i < data_size; ++i) {
    input_data[i] = distribution(generator);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  auto bubbleSort = std::make_shared<kapustin_i_bubble_sort_seq::BubbleSortSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bubbleSort);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(input_data.begin(), input_data.end());

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], data_size * sizeof(int));

  ASSERT_EQ(output_data, input_data);
}
