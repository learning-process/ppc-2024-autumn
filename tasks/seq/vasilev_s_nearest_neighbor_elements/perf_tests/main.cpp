#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vasilev_s_nearest_neighbor_elements/include/ops_seq.hpp"

TEST(vasilev_s_nearest_neighbor_elements_seq, test_pipeline_run) {
  std::vector<int> input_vec;
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 100000;

  input_vec.resize(count_size_vector);
  for (int i = 0; i < count_size_vector; ++i) {
    input_vec[i] = i;
  }
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  auto taskSequential =
      std::make_shared<vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential>(taskDataSeq);
  ASSERT_EQ(taskSequential->validation(), true);
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

  int expected_min_diff = 1;
  int expected_index1 = 0;
  int expected_index2 = 1;
  ASSERT_EQ(output_result[0], expected_min_diff);
  ASSERT_EQ(output_result[1], expected_index1);
  ASSERT_EQ(output_result[2], expected_index2);
}

TEST(vasilev_s_nearest_neighbor_elements_seq, test_task_run) {
  std::vector<int> input_vec;
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 100000;

  input_vec.resize(count_size_vector);
  for (int i = 0; i < count_size_vector; ++i) {
    input_vec[i] = i;
  }
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  auto taskSequential =
      std::make_shared<vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential>(taskDataSeq);
  ASSERT_EQ(taskSequential->validation(), true);
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

  int expected_min_diff = 1;
  int expected_index1 = 0;
  int expected_index2 = 1;
  ASSERT_EQ(output_result[0], expected_min_diff);
  ASSERT_EQ(output_result[1], expected_index1);
  ASSERT_EQ(output_result[2], expected_index2);
}
