#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/vavilov_v_contrast_enhancement/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(PreProcessingTest, ValidInput) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    taskData->inputs_count[0] = input.size();
    taskData->inputs[0] = reinterpret_cast<void*>(input.data());

    vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);
    ASSERT_TRUE(task.pre_processing());
  }
}

TEST(ValidationTest, CorrectOutputSize) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count[0] = 10;
    taskData->outputs_count[0] = 10;

    vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.validation());
  }
}

TEST(ValidationTest, IncorrectOutputSize) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count[0] = 10;
    taskData->outputs_count[0] = 8;

    vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);
    ASSERT_TRUE(task.pre_processing());
    ASSERT_FALSE(task.validation());
  }
}

TEST(RunTest, NormalContrastEnhancement) {
  mpi::environment env;
  mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskData->inputs_count[0] = input.size();
  taskData->outputs_count[0] = input.size();
  std::vector<int> output(input.size());
  taskData->inputs[0] = reinterpret_cast<void*>(input.data());
  taskData->outputs[0] = reinterpret_cast<void*>(output.data());

  vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());

  if (world.rank() == 0) {
    std::vector<int> expected_output = {0, 63, 127, 191, 255};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(PostProcessingTest, ValidOutputCopy) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input = {10, 20, 30, 40, 50};
    taskData->inputs_count[0] = input.size();
    taskData->outputs_count[0] = input.size();
    std::vector<int> output(input.size());
    taskData->inputs[0] = reinterpret_cast<void*>(input.data());
    taskData->outputs[0] = reinterpret_cast<void*>(output.data());

    vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel task(taskData);
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    std::vector<int> expected_output = {0, 63, 127, 191, 255};
    EXPECT_EQ(output, expected_output);
  }
}
