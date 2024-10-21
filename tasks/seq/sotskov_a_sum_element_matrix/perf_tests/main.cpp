// Copyright 2024 Sotskov Andrey
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include "core/perf/include/perf.hpp"
#include "seq/sotskov_a_sum_element_matrix/include/ops_seq.hpp"

TEST(sotskov_a_sum_element_matrix_perf_test, test_pipeline_run) {
    std::vector<std::vector<int>> matrix(100, std::vector<int>(100, 1));
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&matrix));
    taskData->inputs_count.push_back(1);
    int output = 0;
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskData->outputs_count.push_back(1);

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_TRUE(task.validation()); 
    auto start = std::chrono::high_resolution_clock::now();
    task.pre_processing();
    task.run();
    task.post_processing();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time to sum 100x100 matrix: " << diff.count() << " s\n";
    ASSERT_EQ(output, 10000);
}

TEST(sotskov_a_sum_element_matrix_perf_test, test_task_run) {
    std::vector<std::vector<int>> matrix(5000, std::vector<int>(5000, 1));
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&matrix));
    taskData->inputs_count.push_back(1);
    int output = 0;
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskData->outputs_count.push_back(1);

    // Create Task
    sotskov_a_sum_element_matrix_seq::TestTaskSequential task(taskData);
    ASSERT_TRUE(task.validation());
    auto start = std::chrono::high_resolution_clock::now();
    task.pre_processing();
    task.run();
    task.post_processing();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time to sum 5000x5000 matrix: " << diff.count() << " s\n";
    ASSERT_EQ(output, 25000000);
}
