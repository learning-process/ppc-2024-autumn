#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

TEST(SequentialContrastPerf, TestContrastPerfWith1000Elements) {
  std::vector<uint8_t> input_vec(1000, 128);
  std::vector<uint8_t> output_vec(1000, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << "Execution time for 1000 elements: " << duration << " ms" << std::endl;
  for (std::size_t i = 0; i < output_vec.size(); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}

TEST(SequentialContrastPerf, TestContrastPerfWith10000Elements) {
  std::vector<uint8_t> input_vec(10000, 128);
  std::vector<uint8_t> output_vec(10000, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << "Execution time for 10000 elements: " << duration << " ms" << std::endl;
  for (std::size_t i = 0; i < output_vec.size(); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}

TEST(SequentialContrastPerf, TestContrastPerfWith100000Elements) {
  std::vector<uint8_t> input_vec(100000, 128);
  std::vector<uint8_t> output_vec(100000, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << "Execution time for 100000 elements: " << duration << " ms" << std::endl;
  for (std::size_t i = 0; i < output_vec.size(); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}

TEST(SequentialContrastPerf, TestContrastPerfWith1000000Elements) {
  std::vector<uint8_t> input_vec(1000000, 128);
  std::vector<uint8_t> output_vec(1000000, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << "Execution time for 1000000 elements: " << duration << " ms" << std::endl;
  for (std::size_t i = 0; i < output_vec.size(); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}