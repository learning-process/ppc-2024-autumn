// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"

TEST(sedova_o_vertical_ribbon_scheme, Test_Small_Matrix) {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
    std::vector<int> vector = {1, 2};
    std::vector<int> expected_result = {5, 11};
    std::vector<int> result(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result[0], expected_result[0]);
    ASSERT_EQ(result[1], expected_result[1]);
}

TEST(sedova_o_vertical_ribbon_scheme, Test_Larger_Matrix) {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int> vector = {1, 2, 3};
    std::vector<int> expected_result = {14, 32, 50};
    std::vector<int> result(3,0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}


// Add more func_test cases here with varying matrix sizes and values...
TEST(sedova_o_vertical_ribbon_scheme, Test_Zero_Matrix) {
    std::vector<int> matrix = {0, 0, 0, 0};
    std::vector<int> vector = {1, 2};
    std::vector<int> expected_result = {0, 0};
    std::vector<int> result(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);

}
TEST(sedova_o_vertical_ribbon_scheme, Test_Negative_Matrix) {
    std::vector<int> matrix = {-1, -2, -3, -4};
    std::vector<int> vector = {-1, -2};
    std::vector<int> expected_result = {5, 11};
    std::vector<int> result(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);

}
TEST(sedova_o_vertical_ribbon_scheme, Test_3x3_Matrix) {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> vector = {1, 2, 3};
    std::vector<int> expected_result = {14, 32, 50};
    std::vector<int> result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}
TEST(sedova_o_vertical_ribbon_scheme, Test_5x5_Matrix) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  std::vector<int> vector = {1, 2, 3, 4, 5};
  std::vector<int> expected_result = {55, 130, 205, 280, 355};
    std::vector<int> result(5, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme, Test_2x4_Matrix) {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> vector = {1, 2, 3, 4};
    std::vector<int> expected_result = {30, 70};
    std::vector<int> result(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}
TEST(sedova_o_vertical_ribbon_scheme, Test_4x2_Matrix) {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> vector = {1, 2};
    std::vector<int> expected_result = {5, 11, 17, 23};
    std::vector<int> result(4, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}
TEST(sedova_o_vertical_ribbon_scheme, Test_1x1_Matrix) {
    std::vector<int> matrix = {1};
    std::vector<int> vector = {1};
    std::vector<int> expected_result = {1};
    std::vector<int> result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}
TEST(sedova_o_vertical_ribbon_scheme, Test_2x2_Matrix) {
    std::vector<int> matrix = {1, 2, 3, 4};
    std::vector<int> vector = {1, 2};
    std::vector<int> expected_result = {5, 11};
    std::vector<int> result(2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}
TEST(sedova_o_vertical_ribbon_scheme, Test_All_Ones) {
    std::vector<int> matrix(9, 1);
    std::vector<int> vector(3, 1);
    std::vector<int> expected_result(3, 3);
    std::vector<int> result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.push_back(matrix.size());
    taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.push_back(vector.size());
    taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
    taskDataSeq->outputs_count.push_back(result.size());
    sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(result, expected_result);
}
