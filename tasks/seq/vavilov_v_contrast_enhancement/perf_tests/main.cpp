#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vavilov_v_contrast_enhancement/include/ops_seq.hpp"

TEST(vavilov_v_contrast_enhancement_seq, RunTaskWithLargeInput) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t data_size = 1000000;
  std::vector<int> input(data_size, 128);
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs_count.emplace_back(input.size());

  std::vector<int> output(input.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskDataSeq);

  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Large input task run time: " << elapsed.count() << " seconds" << std::endl;
}

TEST(vavilov_v_contrast_enhancement_seq, RunPipelineWithMultipleTasks) {
  size_t num_tasks = 10;
  size_t data_size = 100000;

  for (size_t i = 0; i < num_tasks; ++i) {
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input(data_size, i);
    std::vector<int> output(data_size);

    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

    vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskDataSeq);
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());
  }
}
