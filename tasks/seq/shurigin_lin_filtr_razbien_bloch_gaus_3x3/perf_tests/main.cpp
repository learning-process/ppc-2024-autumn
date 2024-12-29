#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shurigin_lin_filtr_razbien_bloch_gaus_3x3/include/ops_seq.hpp"

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, Performance_Pipeline_Run) {
  const int size = 500;
  std::vector<double> input(size * size, 1.0);
  std::vector<double> output(size * size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input.data())));
  task_data->inputs_count.push_back(input.size());

  int rows = size;
  int cols = size;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(output.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [start]() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->pipeline_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  std::vector<double> expected(size * size, 1.0);
  for (int i = 0; i < size; ++i) {
    expected[i] = 0.0;
    expected[(size - 1) * size + i] = 0.0;
    if (i > 0 && i < size - 1) {
      expected[i * size] = 0.0;
      expected[i * size + size - 1] = 0.0;
    }
  }
  EXPECT_EQ(output, expected);
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_seq, Performance_Task_Run) {
  const int size = 500;
  std::vector<double> input(size * size, 1.0);
  std::vector<double> output(size * size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input.data())));
  task_data->inputs_count.push_back(input.size());

  int rows = size;
  int cols = size;
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.push_back(1);

  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(output.size());

  auto task = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq>(task_data);
  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [start]() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->task_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  std::vector<double> expected(size * size, 1.0);
  for (int i = 0; i < size; ++i) {
    expected[i] = 0.0;
    expected[(size - 1) * size + i] = 0.0;
    if (i > 0 && i < size - 1) {
      expected[i * size] = 0.0;
      expected[i * size + size - 1] = 0.0;
    }
  }
  EXPECT_EQ(output, expected);
}