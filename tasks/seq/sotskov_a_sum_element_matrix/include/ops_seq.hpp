#pragma once

#include <vector>
#include <memory>

#include "core/task/include/task.hpp"

namespace sotskov_a_sum_element_matrix_seq {

std::vector<int> create_random_matrix_int(int rows, int cols);
std::vector<double> create_random_matrix_double(int rows, int cols);
int sum_matrix_elements_int(const std::vector<int>& matrix, int rows, int cols);
double sum_matrix_elements_double(const std::vector<double>& matrix, int rows, int cols);
int random_range(int min, int max);

class TestTaskSequentialInt : public ppc::core::Task {
public:
    explicit TestTaskSequentialInt(std::shared_ptr<ppc::core::TaskData> task_data);
    bool pre_processing() override;
    bool validation() override;
    bool run() override;
    bool post_processing() override;

private:
    std::vector<int> input_data_;
    int result_{0};
};

class TestTaskSequentialDouble : public ppc::core::Task {
public:
    explicit TestTaskSequentialDouble(std::shared_ptr<ppc::core::TaskData> task_data);
    bool pre_processing() override;
    bool validation() override;
    bool run() override;
    bool post_processing() override;

private:
    std::vector<double> input_data_;
    double result_{0.0};
};

}  // namespace sotskov_a_sum_element_matrix_seq
