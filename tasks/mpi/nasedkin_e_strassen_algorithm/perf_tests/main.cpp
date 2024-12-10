#include <gtest/gtest.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

std::vector<double> generate_dense_matrix(size_t n, double min_val = 1.0, double max_val = 10.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);
    std::vector<double> matrix(n * n);
    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

void measure_performance(size_t n) {
    boost::mpi::communicator world;
    boost::mpi::timer timer;

    auto matrix_a = generate_dense_matrix(n);
    auto matrix_b = generate_dense_matrix(n);

    auto taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<double> result_matrix(n * n, 0);

    if (world.rank() == 0) {
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
        taskData->inputs_count.push_back(n);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_b.data()));
        taskData->inputs_count.push_back(n);
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_matrix.data()));
        taskData->outputs_count.push_back(n);
    }

    nasedkin_e_strassen_algorithm::TestMPITaskParallel task(taskData);

    ASSERT_TRUE(task.validation());
    task.pre_processing();
    double start_time = timer.elapsed();
    task.run();
    double elapsed_time = timer.elapsed() - start_time;
    task.post_processing();

    if (world.rank() == 0) {
        std::cout << "Matrix size: " << n << "x" << n << " | Elapsed time: " << elapsed_time << " seconds\n";
    }
}

TEST(MPI_Strassen_Perf, Perf_64x64) {
    measure_performance(64);
}

TEST(MPI_Strassen_Perf, Perf_128x128) {
    measure_performance(128);
}

TEST(MPI_Strassen_Perf, Perf_256x256) {
    measure_performance(256);
}

