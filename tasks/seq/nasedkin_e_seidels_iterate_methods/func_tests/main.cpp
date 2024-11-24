#include <gtest/gtest.h>
#include "seq/nasedkin_e_seidels_iterate_methods/include/ops_seq.hpp"

TEST(nasedkin_e_seidels_iterate_methods_seq, test_with_valid_input) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(3);

    nasedkin_e_seidels_iterate_methods_seq::SeidelIterateMethodsSeq seidel_task(taskData);

    ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid input";
    ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed";
    ASSERT_TRUE(seidel_task.run()) << "Run failed";
    ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed";
}

TEST(nasedkin_e_seidels_iterate_methods_seq, test_with_invalid_input) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(0);

    nasedkin_e_seidels_iterate_methods_seq::SeidelIterateMethodsSeq seidel_task(taskData);

    ASSERT_FALSE(seidel_task.validation());
}
