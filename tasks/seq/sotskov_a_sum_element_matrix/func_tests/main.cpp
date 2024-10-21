#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "seq/sotskov_a_sum_element_matrix/include/ops_seq.hpp"

namespace sotskov_a_sum_element_matrix_seq {

int random_range(int min, int max) {
    return min + rand() % (max - min + 1);
}

template <typename T>
void run_sum_test(const std::vector<T>& matrix, std::vector<typename std::conditional<std::is_same<T, double>::value, double, int32_t>::type>& reference_sum, int rows, int columns) {
    reference_sum[0] = sum_matrix_elements(matrix, rows, columns);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<T*>(matrix.data())));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    TestTaskSequential<T> testTask(taskDataSeq);
    ASSERT_TRUE(testTask.validation());
    testTask.pre_processing();
    testTask.run();
    testTask.post_processing();

    ASSERT_EQ(reference_sum[0], sum_matrix_elements(matrix, rows, columns));
}

TEST(sotskov_a_sum_element_matrix, check_large_matrix) {
    const int rows = 1000;
    const int columns = 1000;

    std::vector<double> global_matrix = create_random_matrix<double>(rows, columns);
    std::vector<double> reference_sum(1, 0);

    run_sum_test(global_matrix, reference_sum, rows, columns);
}

TEST(sotskov_a_sum_element_matrix, check_negative_value) {
    const int rows = 10;
    const int columns = 10;

    std::vector<int> global_matrix = create_random_matrix<int>(rows, columns);
    for (auto& elem : global_matrix) {
        elem = -abs(elem);
    }
    std::vector<int32_t> reference_sum(1, 0);

    run_sum_test(global_matrix, reference_sum, rows, columns);
}

TEST(sotskov_a_sum_element_matrix, check_int) {
    srand(static_cast<unsigned int>(time(0)));

    const int rows = random_range(0, 100);
    const int columns = random_range(0, 100);

    std::vector<int> global_matrix = create_random_matrix<int>(rows, columns);
    std::vector<int32_t> reference_sum(1, 0);
    
    run_sum_test(global_matrix, reference_sum, rows, columns);
}

TEST(sotskov_a_sum_element_matrix, check_double) {
    srand(static_cast<unsigned int>(time(0)));

    const int rows = random_range(0, 100);
    const int columns = random_range(0, 100);

    std::vector<double> global_matrix = create_random_matrix<double>(rows, columns);
    std::vector<double> reference_sum(1, 0.0);

    run_sum_test(global_matrix, reference_sum, rows, columns);
}

TEST(sotskov_a_sum_element_matrix, check_empty) {
    std::vector<int32_t> reference_sum(1, 0);
    std::vector<int> empty_matrix;

    run_sum_test(empty_matrix, reference_sum, 0, 0);
}

TEST(sotskov_a_sum_element_matrix, check_zero) {
    auto zero_columns = create_random_matrix<int>(1, 0);
    EXPECT_TRUE(zero_columns.empty());
    auto zero_rows = create_random_matrix<int>(0, 1);
    EXPECT_TRUE(zero_rows.empty());
}

TEST(sotskov_a_sum_element_matrix, check_wrong_valid) {
    std::vector<int> global_matrix;
    std::vector<int32_t> global_sum(2, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataSeq->outputs_count.emplace_back(global_sum.size());

    TestTaskSequential<int> testTask(taskDataSeq);
    ASSERT_FALSE(testTask.validation());
}

}  // namespace sotskov_a_sum_element_matrix_seq
