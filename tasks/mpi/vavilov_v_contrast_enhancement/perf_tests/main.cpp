#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vavilov_v_contrast_enhancement/include/ops_mpi.hpp"

TEST(PerformanceTest, RunLargeInput) {
  mpi::environment env;
  mpi::communicator world;

  size_t data_size = 1000000;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input(data_size, 128);
    taskData->inputs_count[0] = data_size;
    taskData->outputs_count[0] = data_size;

    std::vector<int> output(data_size);
    taskData->inputs[0] = reinterpret_cast<void*>(input.data());
    taskData->outputs[0] = reinterpret_cast<void*>(output.data());

    vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Large input task run time: " << elapsed.count() << " seconds" << std::endl;
  }
}

TEST(PerformanceTest, RunPipelineWithMultipleTasks) {
  mpi::environment env;
  mpi::communicator world;

  size_t num_tasks = 5;
  size_t data_size = 100000;

  for (size_t i = 0; i < num_tasks; ++i) {
    if (world.rank() == 0) {
      auto taskData = std::make_shared<ppc::core::TaskData>();
      std::vector<int> input(data_size, i + 1);
      std::vector<int> output(data_size);

      taskData->inputs_count[0] = data_size;
      taskData->outputs_count[0] = data_size;
      taskData->inputs[0] = reinterpret_cast<void*>(input.data());
      taskData->outputs[0] = reinterpret_cast<void*>(output.data());

      vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);
      ASSERT_TRUE(task.pre_processing());
      ASSERT_TRUE(task.validation());
      ASSERT_TRUE(task.run());
      ASSERT_TRUE(task.post_processing());
    }
  }
}
