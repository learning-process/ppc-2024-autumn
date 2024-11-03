#include "mpi/laganina_e_sum_values_by_columns_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> laganina_e_sum_values_by_columns_matrix_mpi::getRandomVector(int sz) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) {
        vec[i] = (gen() % 100) - 49;
    }
    return vec;
}

std::vector<int> laganina_e_sum_values_by_columns_matrix_mpi::SumSeq(const std::vector<int>& matrix, int n, int m,
    int x0, int x1) {
    std::vector<int> result;
    for (int j = x0; j< x1; j++) {
        int sum = 0;
        for (int i= 0; i < m; i++) {
            sum += matrix[i * n + j];
        }
        result[j]=sum;
    }
    return result;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::pre_processing() {
    internal_order_test();
    input_ = std::vector<int>(taskData->inputs_count[0]);
    m = taskData->inputs_count[1];
    n = taskData->inputs_count[2];
    auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
        input_[i] = ptr[i];
    }
    res_ = std::vector<int>(n, 0);
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::validation() {
    internal_order_test();
    if (taskData->inputs_count[2] != taskData->outputs_count[0]) {
        return false;
    };
    if (taskData->inputs_count[1] < 1 || taskData->inputs_count[2] < 1) {
        return false;
    }
    if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[2]) {
        return false;
    }
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::run() {
    internal_order_test();
    for (int j = 0; j < n; j++) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            sum += input_[i * n + j];
        }
        res_[j] = sum;
    }
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::post_processing() {
    internal_order_test();
    for (int i = 0; i < n; i++) {
        reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
    }
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::pre_processing() {
    internal_order_test();
    if (world.rank() == 0) {
        m= taskData->inputs_count[1];
        n= taskData->inputs_count[2];
    }
    broadcast(world, n, 0);
    broadcast(world, m, 0);
    // fbd nt ncssr dlt
    if (world.rank() == 0) {
        // Init vectors
        input_ = std::vector<int>(taskData->inputs_count[0]);
        auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
        for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
            input_[i] = tmp_ptr[i];
        }
    }
    else {
        input_ = std::vector<int>(n * m);
    }
    broadcast(world, input_.data(), n * m, 0);
    // Init value for output
    res_ = std::vector<int>(m, 0);
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::validation() {
    internal_order_test();
    if (world.rank() == 0) {
        if (taskData->inputs_count[2] != taskData->outputs_count[0]) {
            return false;
        };
        if (taskData->inputs_count[1] < 1 || taskData->inputs_count[2] < 1) {
            return false;
        }
        if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[2]) {
            return false;
        }
        return true;
        
    }
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::run() {
    internal_order_test();
    int delta = n/ world.size();
    delta += (n % world.size() == 0) ? 0 : 1;
    int lastColumn = std::min(n, delta * (world.rank() + 1));
    auto partSum = SumSeq(input_, n, m, delta * world.rank(), lastColumn);
    partSum.resize(delta);
    if (world.rank() == 0) {
        std::vector<int> partRes(n + delta * world.size());
        std::vector<int> sizes(world.size(), delta);
        boost::mpi::gatherv(world, partSum.data(), partSum.size(), partRes.data(), sizes, 0);
        partRes.resize(n);
        res_ = partRes;
    }
    else {
        boost::mpi::gatherv(world, partSum.data(), partSum.size(), 0);
    }
    return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::post_processing() {
    internal_order_test();
    if (world.rank() == 0) {
        for (int i = 0; i < n; i++) {
            reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
        }
    }
    return true;
}