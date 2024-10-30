// Copyright 2024 Alputov Ivan
#include <gtest/gtest.h>

#include <utility>
#include <vector>
#include <random> 
#include "seq/alputov_i_most_diff_neighbor_elem/include/ops_seq.hpp"

// Test for maximum difference in a typical case
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_Typical) {
  std::vector<int> inputVector = {10, 20, 40, 80, 128, 78, -12, -15, 44, 90, 51};
  std::pair<int, int> expectedResult = {78, -12}; 


  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());


  std::vector<std::pair<int, int>> outputPairs(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPairs.data()));
  taskDataSeq->outputs_count.emplace_back(outputPairs.size());

  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());  
  testTaskSequential.pre_processing();           
  testTaskSequential.run();                      
  testTaskSequential.post_processing();         

  ASSERT_EQ(outputPairs[0], expectedResult);  
}

// Test for maximum difference with negative values
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_NegativeValues) {
  std::vector<int> inputVector = {-3, -6, -9, -11};
  std::pair<int, int> expectedResult = {-6, -9};  // Adjust according to the expected max difference pair

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  std::pair<int, int> outputPair = {0, 0};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outputPair));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair, expectedResult);
}

// Test for maximum difference with equal elements
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_EqualElements) {
  std::vector<int> inputVector = {2, 2, 2, 2, 2};
  std::pair<int, int> expectedResult = {2, 2};  // Adjust according to the expected max difference pair

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  std::pair<int, int> outputPair = {0, 0};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outputPair));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair, expectedResult);
}

// Test for maximum difference in a two-element case
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_TwoElements) {
  std::vector<int> inputVector = {5, 10};
  std::pair<int, int> expectedResult = {5, 10};  // Adjust according to the expected max difference pair

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  std::pair<int, int> outputPair = {0, 0};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outputPair));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair, expectedResult);
}

// Test for a single element vector
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_SingleElement) {
  std::vector<int> inputVector = {100};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  std::pair<int, int> outputPair = {0, 0};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outputPair));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

// Test for an empty vector
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_EmptyVector) {
  std::vector<int> inputVector = {};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  std::pair<int, int> outputPair = {0, 0};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outputPair));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_RandomLargeVector) {
  const int count = 1000000;  
  const int fixedSeed = 12345;  
  std::mt19937 gen(fixedSeed);  
  std::uniform_int_distribution<> dist(-1000, 1000);  

  std::vector<int> inputVector(count);
  for (int i = 0; i < count; ++i) {
    inputVector[i] = dist(gen); 
  }

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  std::pair<int, int> outputPair = {0, 0};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outputPair));
  taskDataSeq->outputs_count.emplace_back(1);

  // Создаем задачу
  alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::pair<int, int> expectedResult = {1000, -1000}; 

  ASSERT_EQ(outputPair, expectedResult);
}