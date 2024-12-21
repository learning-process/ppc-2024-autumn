#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

TEST(mezhuev_m_most_different_neighbor_elements, LargeDataset) {
  const size_t large_size = 1000000;
  std::vector<int> input_data(large_size);
  for (size_t i = 0; i < large_size; ++i) {
    input_data[i] = static_cast<int>(i % 1000 - 500);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(2, 0);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskData->outputs_count.push_back(static_cast<uint32_t>(output_data.size()));

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);

  ASSERT_TRUE(task.validation()) << "Validation failed for large dataset.";

  ASSERT_TRUE(task.pre_processing()) << "Pre-processing failed for large dataset.";
  auto start_time = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(task.run()) << "Run failed for large dataset.";
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Time taken for `run` with large dataset: " << elapsed_time.count() << " seconds" << std::endl;

  ASSERT_TRUE(task.post_processing()) << "Post-processing failed for large dataset.";

  ASSERT_NE(output_data[0], output_data[1]) << "Output values should not be the same for large dataset.";
}

TEST(mezhuev_m_most_different_neighbor_elements, EdgeCaseSingleElement) {
  std::vector<int> input_data = {42};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(2, 0);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskData->outputs_count.push_back(static_cast<uint32_t>(output_data.size()));

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);

  ASSERT_FALSE(task.validation()) << "Validation should fail for single element.";
}

TEST(mezhuev_m_most_different_neighbor_elements, EdgeCaseEmptyInput) {
  std::vector<int> input_data;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(2, 0);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskData->outputs_count.push_back(static_cast<uint32_t>(output_data.size()));

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(taskData);

  ASSERT_FALSE(task.validation()) << "Validation should fail for empty input.";
}