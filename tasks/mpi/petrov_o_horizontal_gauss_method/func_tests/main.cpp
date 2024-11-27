#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <numeric>
#include <iostream>
#include <boost/mpi.hpp>
#include <random>

#include "mpi/petrov_o_horizontal_gauss_method/include/ops_mpi.hpp"

// Вспомогательная функция для генерации случайной матрицы
std::vector<std::vector<double>> generateRandomMatrix(size_t n, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min_val, max_val);

    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[i][j] = dist(gen);
        }
    }
    return matrix;
}

// Вспомогательная функция для генерации случайного вектора
std::vector<double> generateRandomVector(size_t n, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min_val, max_val);

    std::vector<double> vec(n);
    for (size_t i = 0; i < n; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}



TEST(petrov_o_horizontal_gauss_method_par, TestGauss_Simple) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    size_t n = 3;
    std::vector<double> input_matrix = {2, 1, 0, 
                                        -3, -1, 2,
                                        0, 1, 2};
    std::vector<double> input_b = {8, -11, -3};
    std::vector<double> output(n);

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
        taskData->inputs_count.push_back(n);
        taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
        taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_b.data()));
        taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
        taskData->outputs_count.push_back(n * sizeof(double));
    }

    petrov_o_horizontal_gauss_method_mpi::ParallelTask task(taskData);

    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (world.rank() == 0) {
        ASSERT_DOUBLE_EQ(output[0], 8);
        ASSERT_DOUBLE_EQ(output[1], -8);
        ASSERT_DOUBLE_EQ(output[2], 2.5);
    }
}


// Тест для параллельной версии с рандомной матрицей
// TEST(petrov_o_horizontal_gauss_method_par, TestGauss_RandomMatrix) {
//     boost::mpi::communicator world;
    
//     size_t n = 10; // Размер матрицы (можно менять для тестирования)

//     std::vector<std::vector<double>> random_matrix;
//     std::vector<double> random_b;
//     std::vector<double> par_output(n);

//      if (world.rank() == 0) {
//                 random_matrix = generateRandomMatrix(n,-100,100);
//                 random_b = generateRandomVector(n, -100, 100);
//      }



//     std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
//      if (world.rank() == 0) {
//                 taskData->inputs_count.push_back(n);
//                 taskData->inputs.push_back(reinterpret_cast<uint8_t*>(random_matrix.data()));
//                 taskData->inputs.push_back(reinterpret_cast<uint8_t*>(random_b.data()));
//                 taskData->outputs.push_back(reinterpret_cast<uint8_t*>(par_output.data()));
//                 taskData->outputs_count.push_back(n * sizeof(double));
//     }

//     petrov_o_horizontal_gauss_method_mpi::ParallelTask task(taskData);

//     ASSERT_TRUE(task.validation());
//     ASSERT_TRUE(task.pre_processing());
//     ASSERT_TRUE(task.run());
//     ASSERT_TRUE(task.post_processing());
// }


// ... (Другие тесты, адаптированные из последовательной версии)