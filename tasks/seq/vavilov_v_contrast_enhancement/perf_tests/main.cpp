#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vavilov_v_contrast_enhancement/include/ops_seq.hpp"

TEST(vavilov_v_contrast_enhancement_seq, RunTaskWithLargeInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  size_t data_size = 1000000;
  std::vector<int> input(data_size, 128);
  taskData->inputs_count[0] = data_size;
  taskData->outputs_count[0] = data_size;

  std::vector<int> output(data_size);
  taskData->inputs[0] = reinterpret_cast<void*>(input.data());
  taskData->outputs[0] = reinterpret_cast<void*>(output.data());

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);

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
    auto taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input(data_size, i);
    std::vector<int> output(data_size);

    taskData->inputs_count[0] = data_size;
    taskData->outputs_count[0] = data_size;
    taskData->inputs[0] = reinterpret_cast<void*>(input.data());
    taskData->outputs[0] = reinterpret_cast<void*>(output.data());

    vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());
  }
}
