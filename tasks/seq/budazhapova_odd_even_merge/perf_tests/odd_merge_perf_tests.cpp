#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/budazhapova_odd_even_merge/include/odd_even_merge.hpp"

namespace budazhapova_betcher_odd_even_merge_seq {
std::vector<int> generateRandomVector(int size, int minValue, int maxValue) {
  std::vector<int> randomVector;
  randomVector.reserve(size);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < size; ++i) {
    int randomNum = std::rand() % (maxValue - minValue + 1) + minValue;
    randomVector.push_back(randomNum);
  }
  return randomVector;
}

TEST(budazhapova_betcher_odd_even_merge_seq, test_pipeline_run) {
  std::vector<int> input_vector = generateRandomVector(10000000, 5, 100);
  std::vector<int> out(10000000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<budazhapova_betcher_odd_even_merge_seq::MergeSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
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
}

TEST(budazhapova_betcher_odd_even_merge_seq, test_task_run) {
  std::vector<int> input_vector = generateRandomVector(10000000, 5, 100);
  std::vector<int> out(10000000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<budazhapova_betcher_odd_even_merge_seq::MergeSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
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
}
