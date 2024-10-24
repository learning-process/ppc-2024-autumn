// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "seq/sotskov_a_sum_element_matrix/include/ops_seq.hpp"

TEST(Sequential, Test_Sum_Large_Matrix) {
    const int rows = 1000;
    const int columns = 1000;

    std::vector<double> global_matrix = sotskov_a_sum_element_matrix_seq::create_random_matrix_double(rows, columns);
    std::vector<double> reference_sum(1, 0);

    reference_sum[0] = sotskov_a_sum_element_matrix_seq::sum_matrix_elements_double(global_matrix);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(global_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble testTask(taskDataSeq);
    ASSERT_TRUE(testTask.validation());
    testTask.pre_processing();
    testTask.run();
    testTask.post_processing();
    
    ASSERT_EQ(reference_sum[0], sotskov_a_sum_element_matrix_seq::sum_matrix_elements_double(global_matrix));
}

TEST(Sequential, Test_Sum_Negative_Values) {
    const int rows = 10;
    const int columns = 10;

    std::vector<int> global_matrix = sotskov_a_sum_element_matrix_seq::create_random_matrix_int(rows, columns);
    for (auto& elem : global_matrix) {
        elem = -abs(elem);
    }
    std::vector<int32_t> reference_sum(1, 0);
    reference_sum[0] = sotskov_a_sum_element_matrix_seq::sum_matrix_elements_int(global_matrix);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(global_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt testTask(taskDataSeq);
    ASSERT_TRUE(testTask.validation());
    testTask.pre_processing();
    testTask.run();
    testTask.post_processing();
    
    ASSERT_EQ(reference_sum[0], sotskov_a_sum_element_matrix_seq::sum_matrix_elements_int(global_matrix));
}

TEST(Sequential, Test_Sum_Int) {
    srand(static_cast<unsigned int>(time(nullptr)));

    const int rows = sotskov_a_sum_element_matrix_seq::random_range(1, 100);
    const int columns = sotskov_a_sum_element_matrix_seq::random_range(1, 100);

    std::vector<int> global_matrix = sotskov_a_sum_element_matrix_seq::create_random_matrix_int(rows, columns);
    std::vector<int32_t> reference_sum(1, 0);
    reference_sum[0] = sotskov_a_sum_element_matrix_seq::sum_matrix_elements_int(global_matrix);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(global_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt testTask(taskDataSeq);
    ASSERT_TRUE(testTask.validation());
    testTask.pre_processing();
    testTask.run();
    testTask.post_processing();
    
    ASSERT_EQ(reference_sum[0], sotskov_a_sum_element_matrix_seq::sum_matrix_elements_int(global_matrix));
}

TEST(Sequential, Test_Sum_Double) {
    srand(static_cast<unsigned int>(time(nullptr)));

    const int rows = sotskov_a_sum_element_matrix_seq::random_range(1, 100);
    const int columns = sotskov_a_sum_element_matrix_seq::random_range(1, 100);

    std::vector<double> global_matrix = sotskov_a_sum_element_matrix_seq::create_random_matrix_double(rows, columns);
    std::vector<double> reference_sum(1, 0.0);
    reference_sum[0] = sotskov_a_sum_element_matrix_seq::sum_matrix_elements_double(global_matrix);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(global_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble testTask(taskDataSeq);
    ASSERT_TRUE(testTask.validation());
    testTask.pre_processing();
    testTask.run();
    testTask.post_processing();
    
    ASSERT_EQ(reference_sum[0], sotskov_a_sum_element_matrix_seq::sum_matrix_elements_double(global_matrix));
}

TEST(Sequential, Test_Empty_Matrix) {
    std::vector<int32_t> reference_sum(1, 0);
    std::vector<int> empty_matrix;

    reference_sum[0] = sotskov_a_sum_element_matrix_seq::sum_matrix_elements_int(empty_matrix);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(empty_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(empty_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt testTask(taskDataSeq);
    ASSERT_TRUE(testTask.validation());
    testTask.pre_processing();
    testTask.run();
    testTask.post_processing();
    
    ASSERT_EQ(reference_sum[0], 0);
}

TEST(Sequential, Test_Zero_Columns_Rows) {
    auto zero_columns = sotskov_a_sum_element_matrix_seq::create_random_matrix_int(1, 0);
    EXPECT_TRUE(zero_columns.empty());
    auto zero_rows = sotskov_a_sum_element_matrix_seq::create_random_matrix_int(0, 1);
    EXPECT_TRUE(zero_rows.empty());
}

TEST(Sequential, Test_Wrong_Validation) {
    std::vector<int> global_matrix;
    std::vector<int32_t> global_sum(2, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataSeq->outputs_count.emplace_back(global_sum.size());

    sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt testTask(taskDataSeq);
    ASSERT_FALSE(testTask.validation());
}
