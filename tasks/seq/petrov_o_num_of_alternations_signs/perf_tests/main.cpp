#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/petrov_o_num_of_alternations_signs/include/ops_seq.hpp"

TEST(petrov_o_num_of_alternations_signs_seq, test_pipeline_run) {
  const int size = 100000;  // Большой размер вектора для теста производительности
  std::vector<int> in(size);
  std::iota(in.begin(), in.end(), 1);
  for (size_t i = 0; i < in.size(); ++i) {
    if (i % 2 != 0) {
      in[i] *= -1;
    }
  }
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<petrov_o_num_of_alternations_signs_seq::TestTaskSequential>(taskDataSeq);

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
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out[0],
            static_cast<int>(in.size() -
                             1));  // Проверка на ожидаемое количество чередований (size - 1 для чередующихся знаков)
}

TEST(petrov_o_num_of_alternations_signs_seq, test_task_run) {
  const int size = 100000;  // Большой размер вектора для теста производительности
  std::vector<int> in(size);
  std::iota(in.begin(), in.end(), 1);
  for (size_t i = 0; i < in.size(); ++i) {
    if (i % 2 != 0) {
      in[i] *= -1;
    }
  }
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<petrov_o_num_of_alternations_signs_seq::TestTaskSequential>(taskDataSeq);

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
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out[0],
            static_cast<int>(in.size() -
                             1));  // Проверка на ожидаемое количество чередований (size - 1 для чередующихся знаков)
}
