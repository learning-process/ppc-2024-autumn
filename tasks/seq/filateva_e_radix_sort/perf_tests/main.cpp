// Filateva Elizaveta Radix Sort
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/filateva_e_radix_sort/include/ops_seq.hpp"

namespace filateva_e_radix_sort_seq {

void GeneratorVector(std::vector<int> &vec) {
  int max_z = 100000;
  int min_z = -100000;
  std::random_device dev;
  std::mt19937 gen(dev());
  for (unsigned long i = 0; i < vec.size(); i++) {
    vec[i] = gen() % (max_z - min_z + 1) + min_z;
  }
}

}  // namespace filateva_e_radix_sort_seq

TEST(filateva_e_radix_sort_seq, test_pipeline_run) {
  int size = 400000;
  std::vector<int> vec(size);
  std::vector<int> answer(size);
  std::vector<int> tResh;

  filateva_e_radix_sort_seq::GeneratorVector(vec);
  tResh = vec;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  auto radixSort = std::make_shared<filateva_e_radix_sort_seq::RadixSort>(taskData);

  ASSERT_TRUE(radixSort->validation());
  radixSort->pre_processing();
  radixSort->run();
  radixSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSort);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(tResh.begin(), tResh.end());

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(answer[i], tResh[i]);
  }
}

TEST(filateva_e_radix_sort_seq, test_task_run) {
  int size = 400000;
  std::vector<int> vec(size);
  std::vector<int> answer(size);
  std::vector<int> tResh;

  filateva_e_radix_sort_seq::GeneratorVector(vec);
  tResh = vec;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  auto radixSort = std::make_shared<filateva_e_radix_sort_seq::RadixSort>(taskData);

  ASSERT_TRUE(radixSort->validation());
  radixSort->pre_processing();
  radixSort->run();
  radixSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSort);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(tResh.begin(), tResh.end());

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(answer[i], tResh[i]);
  }
}
