#include <gtest/gtest.h>

#include <vector>

#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

TEST(komshina_d_min_of_vector_elements_seq, Unique_Min_Test_1) {
  const int size = 15000;
  const int expected_min = 1;

 
  std::vector<int> input(size);
  std::vector<int> output(1);
  for (int i = 0; i < size; ++i) {
    input[i] = i + 1;  
  }

 
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

 
  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), true);
  minVectorTask.pre_processing();
  minVectorTask.run();
  minVectorTask.post_processing();
  ASSERT_EQ(expected_min, output[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Unique_Min_Test_2) {
  const int size = 2;
  const int expected_min = -500;

  std::vector<int> input(size, -500);
  std::vector<int> output(1);

 
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  
  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), true);
  minVectorTask.pre_processing();
  minVectorTask.run();
  minVectorTask.post_processing();
  ASSERT_EQ(expected_min, output[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Unique_Min_Test_3) {
  constexpr int size = 5000000;
  constexpr int start_val = -1000000;
  constexpr int expected_min = start_val;

  
  std::vector<int> input(size);
  std::vector<int> output(1);
  for (int i = 0, j = start_val; i < size; ++i, j += 5) {
    input[i] = j;
  }

 
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

 
  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), true);
  minVectorTask.pre_processing();
  minVectorTask.run();
  minVectorTask.post_processing();
  ASSERT_EQ(expected_min, output[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Unique_Min_Test_4) {
  constexpr int size = 5000000;
  constexpr int start_val = -1000000;
  constexpr int expected_min = start_val;

  
  std::vector<int> input(size);
  std::vector<int> output(1);
  for (int i = size - 1, j = start_val; i >= 0; --i, j += 3) {
    input[i] = j;
  }

 
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  
  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), true);
  minVectorTask.pre_processing();
  minVectorTask.run();
  minVectorTask.post_processing();
  ASSERT_EQ(expected_min, output[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Unique_Min_Test_5) {
  const int size = 200;
  const int expected_min = 2;

 
  std::vector<int> input(size, INT_MAX);
  std::vector<int> output(1);
  for (int i = 0; i < size; i += 2) {
    input[i] = 2;  
  }


  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

 
  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), true);
  minVectorTask.pre_processing();
  minVectorTask.run();
  minVectorTask.post_processing();
  ASSERT_EQ(expected_min, output[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Invalid_Input_Test_1) {
 
  std::vector<int> input;
  std::vector<int> output(1);

  
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

 
  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), false);
}

TEST(komshina_d_min_of_vector_elements_seq, Invalid_Input_Test_2) {
  
  std::vector<int> input(4, 10);
  std::vector<int> output;

  
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementsSeq minVectorTask(taskData);
  ASSERT_EQ(minVectorTask.validation(), false);
}
