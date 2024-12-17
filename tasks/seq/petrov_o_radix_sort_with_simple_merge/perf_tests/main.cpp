#include <gtest/gtest.h>

#include <algorithm>  // для std::sort
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/petrov_o_radix_sort_with_simple_merge/include/ops_seq.hpp"

TEST(petrov_o_radix_sort_with_simple_merge_seq, test_pipeline_run) {
  // Создание данных
  std::vector<int> in = {1, 3, 7, 5, 6, 10, 12, 14, 15, 2};
  std::vector<int> out(in.size(), 0);  // Размер выходного массива совпадает с входным

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создание задачи
  auto testTaskSequential =
      std::make_shared<petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential>(taskDataSeq);

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация Perf результатов
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_seq, test_task_run) {
  // Создание данных
  std::vector<int> in = {1, 3, 7, 5, 6, 10, 12, 14, 15, 2};
  std::vector<int> out(in.size(), 0);  // Размер выходного массива совпадает с входным

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создание задачи
  auto testTaskSequential =
      std::make_shared<petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential>(taskDataSeq);

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация Perf результатов
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}