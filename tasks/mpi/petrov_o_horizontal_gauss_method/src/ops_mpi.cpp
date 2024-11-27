#include "mpi/petrov_o_horizontal_gauss_method/include/ops_mpi.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <boost/mpi.hpp>

namespace petrov_o_horizontal_gauss_method_mpi {

bool ParallelTask::validation() {
    internal_order_test();

    int rank;

    rank = world.rank();

    if (rank == 0) {
        // Проверка количества входных/выходных данных
        if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
            return false;
        }

        size_t n = taskData->inputs_count[0];

        if (n == 0) { 
            return false;
        }

        // Проверка наличия указателей на данные
        if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
            return false;
        }
    }

    return true;
}

bool ParallelTask::pre_processing() {
    internal_order_test();

    int rank = world.rank();

    if (rank == 0) {
        // Получаем n из первого входного параметра
        size_t n = taskData->inputs_count[0];

        matrix.resize(n, std::vector<double>(n));
        b.resize(n);
        x.resize(n);

        double* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] = matrix_input[i * n + j];
            }
        }

        double* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
        for (size_t i = 0; i < n; ++i) {
            b[i] = b_input[i];
        }
    }

    return true;
}

bool ParallelTask::run() {
    internal_order_test();

    size_t n = matrix.size();
    int rank = world.rank();
    int size = world.size();

    std::cout << world.size() << " " << world.rank() << std::endl;

    //boost::mpi::broadcast(world, n, 0);

    std::cout << "After broadcast " << world.size() << " " << world.rank() << std::endl;

    // Прямой ход
    for (size_t k = 0; k < n - 1; ++k) {

        // Параллелизация прямого хода с помощью Boost.MPI
        for (size_t i = k + 1 + rank; i < n; i += size) {
            double factor = matrix[i][k] / matrix[k][k];
            for (size_t j = k; j < n; ++j) {
                matrix[i][j] -= factor * matrix[k][j];
            }
            b[i] -= factor * b[k];
        }

        // Синхронизация между процессами
        std::cout << "Rank " << world.rank() << " before barrier, k = " << k << ", n = " << n << std::endl;
        world.barrier();
        std::cout << "Rank " << world.rank() << " after barrier, k = " << k << ", n = " << n << std::endl;
    }
    // std::cout << "Rank " << world.rank() <<" before cycle 1" << std::endl;
    world.barrier(); 
    // std::cout << "Rank " << world.rank() <<" after cycle 1" << std::endl;

    // Обратный ход
    if (rank == 0) {
        x[n - 1] = b[n - 1] / matrix[n - 1][n - 1];
        for (int i = n - 2; i >= 0; --i) {
            double sum = b[i];
            for (size_t j = i + 1; j < n; ++j) {
                sum -= matrix[i][j] * x[j];
            }
            x[i] = sum / matrix[i][i];
        }
    }

    world.barrier(); 
    // std::cout << "Rank " << world.rank() <<" after cycle 2" << std::endl;
    // Распространение результатов на все процессы
    boost::mpi::broadcast(world, x, 0);
    world.barrier(); 

    std::cout << "Rank " << world.rank() <<" on end" << std::endl;
    return true;
}

bool ParallelTask::post_processing() {
    internal_order_test();

    double* output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < x.size(); ++i) {
        output[i] = x[i];
    }

    return true;
}

} // namespace petrov_o_horizontal_gauss_method_mpi

