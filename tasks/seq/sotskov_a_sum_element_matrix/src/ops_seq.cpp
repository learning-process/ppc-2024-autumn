#include "seq/sotskov_a_sum_element_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

int sotskov_a_sum_element_matrix_seq::sum_matrix_elements_int(const std::vector<int>& matrix, int rows, int cols) {
    return std::accumulate(matrix.begin(), matrix.end(), 0);
}

double sotskov_a_sum_element_matrix_seq::sum_matrix_elements_double(const std::vector<double>& matrix, int rows, int cols) {
    return std::accumulate(matrix.begin(), matrix.end(), 0.0);
}

int sotskov_a_sum_element_matrix_seq::random_range(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

std::vector<int> sotskov_a_sum_element_matrix_seq::create_random_matrix_int(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return {};
    }

    std::vector<int> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-100, 100);

    std::generate(matrix.begin(), matrix.end(), [&]() { return dis(gen); });
    return matrix;
}

std::vector<double> sotskov_a_sum_element_matrix_seq::create_random_matrix_double(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return {};
    }

    std::vector<double> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    std::generate(matrix.begin(), matrix.end(), [&]() { return dis(gen); });
    return matrix;
}

sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt::TestTaskSequentialInt(std::shared_ptr<ppc::core::TaskData> task_data) 
    : Task(std::move(task_data)) {}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt::pre_processing() {
    internal_order_test();
    result_ = 0;
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_data_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
    return true;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt::validation() {
    internal_order_test();
    return taskData->outputs_count[0] == 1;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt::run() {
    internal_order_test();
    result_ = std::accumulate(input_data_.begin(), input_data_.end(), 0);
    return true;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialInt::post_processing() {
    internal_order_test();
    if (!taskData->outputs.empty() && taskData->outputs[0] != nullptr) {
        reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
        return true;
    }
    return false;
}

sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble::TestTaskSequentialDouble(std::shared_ptr<ppc::core::TaskData> task_data) 
    : Task(std::move(task_data)) {}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble::pre_processing() {
    internal_order_test();
    result_ = 0.0;
    double* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    input_data_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
    return true;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble::validation() {
    internal_order_test();
    return taskData->outputs_count[0] == 1;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble::run() {
    internal_order_test();
    result_ = std::accumulate(input_data_.begin(), input_data_.end(), 0.0);
    return true;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequentialDouble::post_processing() {
    internal_order_test();
    if (!taskData->outputs.empty() && taskData->outputs[0] != nullptr) {
        reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
        return true;
    }
    return false;
}
