#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>
#include <iostream>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/communicator.hpp>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

namespace nasedkin_e_strassen_algorithm {

    bool StrassenAlgorithmMPI::pre_processing() {
        internal_order_test();
        int rank = world.rank();
        if (rank == 0) {
            auto* inputsA = reinterpret_cast<double*>(taskData->inputs[0]);
            auto* inputsB = reinterpret_cast<double*>(taskData->inputs[1]);

            if (inputsA == nullptr || inputsB == nullptr) {
                return false;
            }

            matrixSize = static_cast<size_t>(std::sqrt(taskData->inputs_count[0]));

            inputMatrixA.assign(inputsA, inputsA + matrixSize * matrixSize);
            inputMatrixB.assign(inputsB, inputsB + matrixSize * matrixSize);
            outputMatrix.resize(matrixSize * matrixSize, 0.0);
        }
        return true;
    }

    bool StrassenAlgorithmMPI::validation() {
        internal_order_test();
        int rank = world.rank();
        if (rank == 0) {
            return !taskData->inputs.empty() && taskData->inputs_count[0] == taskData->inputs_count[1] &&
                   matrix_is_square(taskData->inputs_count[0]) && taskData->inputs_count[0] == taskData->outputs_count[0];
        }
        return true;
    }

    bool StrassenAlgorithmMPI::run() {
        internal_order_test();
        outputMatrix = strassen_multiply(inputMatrixA, inputMatrixB, matrixSize);
        return true;
    }

    bool StrassenAlgorithmMPI::post_processing() {
        internal_order_test();
        int rank = world.rank();
        if (rank == 0) {
            auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
            std::copy(outputMatrix.begin(), outputMatrix.end(), outputs);
        }
        return true;
    }

    std::vector<double> matrix_add(const std::vector<double>& matrixA,
                                   const std::vector<double>& matrixB,
                                   size_t size) {
        std::vector<double> result(size * size);
        for (size_t i = 0; i < size * size; ++i) {
            result[i] = matrixA[i] + matrixB[i];
        }
        return result;
    }

    std::vector<double> matrix_subtract(const std::vector<double>& matrixA,
                                        const std::vector<double>& matrixB,
                                        size_t size) {
        std::vector<double> result(size * size);
        for (size_t i = 0; i < size * size; ++i) {
            result[i] = matrixA[i] - matrixB[i];
        }
        return result;
    }

    bool StrassenAlgorithmMPI::power_of_two(size_t number) {
        if (number == 0) {
            return false;
        }
        return (number & (number - 1)) == 0;
    }

    bool StrassenAlgorithmMPI::matrix_is_square(size_t matrixSize) {
        auto sqrt_val = static_cast<size_t>(std::sqrt(matrixSize));

        return sqrt_val * sqrt_val == matrixSize;
    }

    std::vector<double> pad_matrix(const std::vector<double>& matrix, size_t original_size, size_t new_size) {
        std::vector<double> padded_matrix(new_size * new_size, 0.0);
        for (size_t i = 0; i < original_size; ++i) {
            for (size_t j = 0; j < original_size; ++j) {
                padded_matrix[i * new_size + j] = matrix[i * original_size + j];
            }
        }
        return padded_matrix;
    }

    std::vector<double> StrassenAlgorithmMPI::strassen_recursive(const std::vector<double>& matrixA,
                                           const std::vector<double>& matrixB, size_t size) {
        if (size == 1) {
            return {matrixA[0] * matrixB[0]};
        }

        size_t half_size = size / 2;

        std::vector<double> A11(half_size * half_size);
        std::vector<double> A12(half_size * half_size);
        std::vector<double> A21(half_size * half_size);
        std::vector<double> A22(half_size * half_size);

        std::vector<double> B11(half_size * half_size);
        std::vector<double> B12(half_size * half_size);
        std::vector<double> B21(half_size * half_size);
        std::vector<double> B22(half_size * half_size);

        for (size_t i = 0; i < half_size; ++i) {
            for (size_t j = 0; j < half_size; ++j) {
                A11[i * half_size + j] = matrixA[i * size + j];
                A12[i * half_size + j] = matrixA[i * size + j + half_size];
                A21[i * half_size + j] = matrixA[(i + half_size) * size + j];
                A22[i * half_size + j] = matrixA[(i + half_size) * size + j + half_size];

                B11[i * half_size + j] = matrixB[i * size + j];
                B12[i * half_size + j] = matrixB[i * size + j + half_size];
                B21[i * half_size + j] = matrixB[(i + half_size) * size + j];
                B22[i * half_size + j] = matrixB[(i + half_size) * size + j + half_size];
            }
        }

        std::vector<double> M1 = strassen_recursive(matrix_add(A11, A22, half_size),
                                                    matrix_add(B11, B22, half_size), half_size);
        std::vector<double> M2 = strassen_recursive(matrix_add(A21, A22, half_size), B11, half_size);
        std::vector<double> M3 = strassen_recursive(A11, matrix_subtract(B12, B22, half_size), half_size);
        std::vector<double> M4 = strassen_recursive(A22, matrix_subtract(B21, B11, half_size), half_size);
        std::vector<double> M5 = strassen_recursive(matrix_add(A11, A12, half_size), B22, half_size);
        std::vector<double> M6 = strassen_recursive(matrix_subtract(A21, A11, half_size),
                                                    matrix_add(B11, B12, half_size), half_size);
        std::vector<double> M7 = strassen_recursive(matrix_subtract(A12, A22, half_size),
                                                    matrix_add(B21, B22, half_size), half_size);

        std::vector<double> C11 = matrix_add(matrix_subtract(matrix_add(M1, M4, half_size), M5, half_size), M7, half_size);
        std::vector<double> C12 = matrix_add(M3, M5, half_size);
        std::vector<double> C21 = matrix_add(M2, M4, half_size);
        std::vector<double> C22 = matrix_add(matrix_subtract(matrix_add(M1, M3, half_size), M2, half_size), M6, half_size);

        std::vector<double> result(size * size);
        for (size_t i = 0; i < half_size; ++i) {
            for (size_t j = 0; j < half_size; ++j) {
                result[i * size + j] = C11[i * half_size + j];
                result[i * size + j + half_size] = C12[i * half_size + j];
                result[(i + half_size) * size + j] = C21[i * half_size + j];
                result[(i + half_size) * size + j + half_size] = C22[i * half_size + j];
            }
        }

        return result;
    }

    std::vector<double> StrassenAlgorithmMPI::strassen_base(const std::vector<double>& matrixA,
                                      const std::vector<double>& matrixB, size_t size) {
        if (size == 1) {
            return {matrixA[0] * matrixB[0]};
        }

        size_t new_size = 1;
        while (new_size < size) {
            new_size *= 2;
        }

        std::vector<double> paddedA(new_size * new_size, 0.0);
        std::vector<double> paddedB(new_size * new_size, 0.0);
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                paddedA[i * new_size + j] = matrixA[i * size + j];
                paddedB[i * new_size + j] = matrixB[i * size + j];
            }
        }

        std::vector<double> result = strassen_recursive(paddedA, paddedB, new_size);

        std::vector<double> final_result(size * size);
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                final_result[i * size + j] = result[i * new_size + j];
            }
        }

        return final_result;
    }

    std::vector<double> StrassenAlgorithmMPI::strassen_multiply(const std::vector<double>& matrixA,
                                          const std::vector<double>& matrixB, size_t size) {
        boost::mpi::environment env;
        boost::mpi::communicator world;

        int rank = world.rank();
        int num_procs = world.size();

        if (size == 1) {
            return {matrixA[0] * matrixB[0]};
        }

        size_t new_size = 1;
        while (new_size < size) {
            new_size *= 2;
        }

        std::vector<double> paddedA = pad_matrix(matrixA, size, new_size);
        std::vector<double> paddedB = pad_matrix(matrixB, size, new_size);

        size_t half_size = new_size / 2;

        std::vector<double> A11(half_size * half_size);
        std::vector<double> A12(half_size * half_size);
        std::vector<double> A21(half_size * half_size);
        std::vector<double> A22(half_size * half_size);

        std::vector<double> B11(half_size * half_size);
        std::vector<double> B12(half_size * half_size);
        std::vector<double> B21(half_size * half_size);
        std::vector<double> B22(half_size * half_size);

        for (size_t i = 0; i < half_size; ++i) {
            for (size_t j = 0; j < half_size; ++j) {
                A11[i * half_size + j] = paddedA[i * new_size + j];
                A12[i * half_size + j] = paddedA[i * new_size + j + half_size];
                A21[i * half_size + j] = paddedA[(i + half_size) * new_size + j];
                A22[i * half_size + j] = paddedA[(i + half_size) * new_size + j + half_size];

                B11[i * half_size + j] = paddedB[i * new_size + j];
                B12[i * half_size + j] = paddedB[i * new_size + j + half_size];
                B21[i * half_size + j] = paddedB[(i + half_size) * new_size + j];
                B22[i * half_size + j] = paddedB[(i + half_size) * new_size + j + half_size];
            }
        }

        std::vector<std::vector<double>> M(7);
        if (rank == 0) {
            std::vector<std::vector<double>> tasks = {
                    matrix_add(A11, A22, half_size),
                    matrix_add(A21, A22, half_size),
                    A11,
                    A22,
                    matrix_add(A11, A12, half_size),
                    matrix_subtract(A21, A11, half_size),
                    matrix_subtract(A12, A22, half_size)
            };

            std::vector<std::vector<double>> tasksB = {
                    matrix_add(B11, B22, half_size),
                    B11,
                    matrix_subtract(B12, B22, half_size),
                    matrix_subtract(B21, B11, half_size),
                    B22,
                    matrix_add(B11, B12, half_size),
                    matrix_add(B21, B22, half_size)
            };

            for (int i = 0; i < 7; ++i) {
                world.send(i % num_procs, 0, tasks[i]);
                world.send(i % num_procs, 0, tasksB[i]);
            }
        }

        for (int i = 0; i < 7; ++i) {
            if (i % num_procs == rank) {
                std::vector<double> taskA;
                std::vector<double> taskB;
                world.recv(0, 0, taskA);
                world.recv(0, 0, taskB);
                M[i] = strassen_base(taskA, taskB, half_size);
                world.send(0, 0, M[i]);
            }
        }

        if (rank == 0) {
            for (int i = 0; i < 7; ++i) {
                world.recv(i % num_procs, 0, M[i]);
            }

            std::vector<double> C11 = matrix_add(matrix_subtract(matrix_add(M[0], M[3], half_size), M[4], half_size), M[6], half_size);
            std::vector<double> C12 = matrix_add(M[2], M[4], half_size);
            std::vector<double> C21 = matrix_add(M[1], M[3], half_size);
            std::vector<double> C22 = matrix_add(matrix_subtract(matrix_add(M[0], M[2], half_size), M[1], half_size), M[5], half_size);

            std::vector<double> result(size * size);
            for (size_t i = 0; i < half_size; ++i) {
                for (size_t j = 0; j < half_size; ++j) {
                    result[i * size + j] = C11[i * half_size + j];
                    result[i * size + j + half_size] = C12[i * half_size + j];
                    result[(i + half_size) * size + j] = C21[i * half_size + j];
                    result[(i + half_size) * size + j + half_size] = C22[i * half_size + j];
                }
            }

            return result;
        }

        return {};
    }

}  // namespace nasedkin_e_strassen_algorithm