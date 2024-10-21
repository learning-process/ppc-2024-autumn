// Copyright 2024 Sotskov Andrey
#include <gtest/gtest.h>
#include <vector>
#include "seq/sotskov_a_sum_element_matrix/include/ops_seq.hpp"

TEST(sotskov_a_sum_element_matrix_seq, test_sum_3x3_matrix) {

    std::vector<std::vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix)); 
    taskData->inputs_count.emplace_back(1); 
    int output = 0; 
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output)); 
    taskData->outputs_count.emplace_back(1); 

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();
    ASSERT_EQ(output, 45); 
}

TEST(sotskov_a_sum_element_matrix_seq, test_zero_5x5_matrix) {
    std::vector<std::vector<int>> matrix = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}; 
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix)); 
    taskData->inputs_count.emplace_back(1); 
    int output = 0; 
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output)); 
    taskData->outputs_count.emplace_back(1); 

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();
    ASSERT_EQ(output, 0);
}

TEST(sotskov_a_sum_element_matrix_seq, test_negative_3x3_matrix) {
    std::vector<std::vector<int>> matrix = {{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}};
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskData->inputs_count.emplace_back(1); 
    int output = 0; 
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output)); 
    taskData->outputs_count.emplace_back(1); 

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();
    ASSERT_EQ(output, -45); 
}

TEST(sotskov_a_sum_element_matrix_seq, test_variable_values) {
    std::vector<std::vector<int>> matrix = {{1, 2, 3}, {4, -5, 6}, {7, 8, -9}};
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskData->inputs_count.emplace_back(1);
    int output = 0;
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskData->outputs_count.emplace_back(1);

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();
    ASSERT_EQ(output, 17); 
}

TEST(sotskov_a_sum_element_matrix_seq, test_empty_matrix) {
    std::vector<std::vector<int>> matrix; 
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix)); 
    taskData->inputs_count.emplace_back(1); 
    int output = 0; 
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output)); 
    taskData->outputs_count.emplace_back(1); 

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_FALSE(task.validation()); // Ожидаем, что валидация вернет false для пустой матрицы
    // Не запускаем обработку, так как валидация не прошла
}
