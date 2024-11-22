#include <gtest/gtest.h>
#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/src/ops_mpi.cpp"

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_valid_input) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(3);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    ASSERT_TRUE(seidel_task.validation());
    ASSERT_TRUE(seidel_task.pre_processing());
    ASSERT_TRUE(seidel_task.run());
    ASSERT_TRUE(seidel_task.post_processing());
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_invalid_input) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(0);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    ASSERT_FALSE(seidel_task.validation());
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_negative_input) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(-3);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    ASSERT_FALSE(seidel_task.validation());
}