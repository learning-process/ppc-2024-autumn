#include "../include/ops_mpi.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace nasedkin_e_strassen_algorithm {

    std::vector<int> getRandomMatrix(int size) {
        std::random_device dev;
        std::mt19937 gen(dev());
        std::vector<int> matrix(size * size);
        for (int i = 0; i < size * size; i++) {
            matrix[i] = gen() % 100;
        }
        return matrix;
    }

    std::vector<int> matrixMultiply(const std::vector<int>& A, const std::vector<int>& B, int size) {
        std::vector<int> C(size * size, 0);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
        return C;
    }

    bool StrassenMPITaskParallel::pre_processing() {
        if (world.rank() == 0) {
            size = static_cast<int>(sqrt(taskData->inputs_count[0]));
            A.resize(size * size);
            B.resize(size * size);
            C.resize(size * size, 0);
            std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
                      reinterpret_cast<int*>(taskData->inputs[0]) + size * size, A.begin());
            std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
                      reinterpret_cast<int*>(taskData->inputs[1]) + size * size, B.begin());
        }
        boost::mpi::broadcast(world, size, 0);
        return true;
    }

    bool StrassenMPITaskParallel::validation() {
        if (world.rank() == 0) {
            return taskData->inputs_count[0] == taskData->inputs_count[1] &&
                   taskData->inputs_count[0] == size * size &&
                   taskData->outputs_count[0] == size * size;
        }
        return true;
    }

    bool StrassenMPITaskParallel::run() {
        int local_size = size / world.size();
        std::vector<int> local_A(local_size * size), local_B(local_size * size), local_C(local_size * size, 0);

        if (world.rank() == 0) {
            for (int proc = 0; proc < world.size(); proc++) {
                world.send(proc, 0, A.data() + proc * local_size * size, local_size * size);
                world.send(proc, 1, B.data(), size * size);
            }
        }

        world.recv(0, 0, local_A.data(), local_size * size);
        world.recv(0, 1, local_B.data(), size * size);

        for (int i = 0; i < local_size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    local_C[i * size + j] += local_A[i * size + k] * local_B[k * size + j];
                }
            }
        }

        boost::mpi::reduce(world, local_C.data(), local_size * size, C.data(), std::plus<int>(), 0);
        return true;
    }

    bool StrassenMPITaskParallel::post_processing() {
        if (world.rank() == 0) {
            std::copy(C.begin(), C.end(), reinterpret_cast<int*>(taskData->outputs[0]));
        }
        return true;
    }

}  // namespace nasedkin_e_strassen_algorithm