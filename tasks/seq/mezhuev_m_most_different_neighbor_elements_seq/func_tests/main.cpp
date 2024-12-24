#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

TEST(mezhuev_m_most_different_neighbor_elements, ValidationEmptyInputs) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationInputsCountTooSmall) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {42};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationOutputsCountIncorrect) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(1, 0);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskData->outputs_count.push_back(static_cast<uint32_t>(output_data.size()));

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationAllElementsEqual) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {42, 42, 42};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(2, 0);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskData->outputs_count.push_back(static_cast<uint32_t>(output_data.size()));

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, EmptyInputs) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {};
  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  EXPECT_FALSE(task.validation());
  EXPECT_FALSE(task.pre_processing());
}

TEST(mezhuev_m_most_different_neighbor_elements, EmptyInputsCount) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count = {};
  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  EXPECT_FALSE(task.validation());
  EXPECT_FALSE(task.pre_processing());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidInputs) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));
  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  EXPECT_FALSE(task.validation());
  EXPECT_TRUE(task.pre_processing());
}

TEST(mezhuev_m_most_different_neighbor_elements, CheckInputAssignment) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {4, 5, 6};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));
  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  EXPECT_FALSE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  std::vector<int> expected_input = {4, 5, 6};
  EXPECT_EQ(task.getInput(), expected_input);
}

TEST(mezhuev_m_most_different_neighbor_elements, ResultResizedCorrectly) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {7, 8, 9};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));

  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);

  EXPECT_FALSE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  EXPECT_EQ(task.getResult().size(), static_cast<size_t>(2));
}

TEST(mezhuev_m_most_different_neighbor_elements, TwoElements) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, -15};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));
  taskData->outputs_count.push_back(2);

  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  std::vector<int> expected_result = {10, -15};
  EXPECT_EQ(task.getResult(), expected_result);
}

TEST(mezhuev_m_most_different_neighbor_elements, AllElementsEqual) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {5, 5, 5, 5};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));
  taskData->outputs_count.push_back(2);

  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  ASSERT_FALSE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  std::vector<int> expected_result = {5, 5};
  EXPECT_EQ(task.getResult(), expected_result);
}

TEST(mezhuev_m_most_different_neighbor_elements, InsufficientInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {1};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));
  taskData->outputs_count.push_back(2);

  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, EmptyInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input;
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input.size()));
  taskData->outputs_count.push_back(2);

  auto task = mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, PostProcessingEmptyOutputsAfterProcessing) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> input_data = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  taskData->outputs.clear();

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);

  EXPECT_FALSE(task.validation());
  EXPECT_TRUE(task.pre_processing());
  EXPECT_TRUE(task.run());

  EXPECT_FALSE(task.post_processing());
}