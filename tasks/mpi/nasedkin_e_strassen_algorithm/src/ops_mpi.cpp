#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace nasedkin_e_strassen_algorithm {

    TestMPITaskParallel::TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData)
            : taskData(taskData) {}

    bool TestMPITaskParallel::pre_processing() {
        size_t total_rows, total_cols;
        if (world.rank() == 0) {
            total_rows = taskData->inputs_count[0];
            total_cols = taskData->inputs_count[1];
        }
        boost::mpi::broadcast(world, total_rows, 0);
        boost::mpi::broadcast(world, total_cols, 0);

        rows_per_proc = total_rows / world.size();
        cols_per_proc = total_cols;

        local_matrix_a.resize(rows_per_proc * total_cols);
        local_matrix_b.resize(total_rows * cols_per_proc);
        local_matrix_c.resize(rows_per_proc * cols_per_proc, 0);

        if (world.rank() == 0) {
            const double* matrix_a = reinterpret_cast<double*>(taskData->inputs[0]);
            const double* matrix_b = reinterpret_cast<double*>(taskData->inputs[1]);

            for (int i = 1; i < world.size(); ++i) {
                world.send(i, 0, matrix_a + i * rows_per_proc * total_cols, rows_per_proc * total_cols);
                world.send(i, 1, matrix_b, total_rows * cols_per_proc);
            }
            std::copy(matrix_a, matrix_a + rows_per_proc * total_cols, local_matrix_a.begin());
            std::copy(matrix_b, matrix_b + total_rows * cols_per_proc, local_matrix_b.begin());
        } else {
            world.recv(0, 0, local_matrix_a.data(), rows_per_proc * total_cols);
            world.recv(0, 1, local_matrix_b.data(), rows_per_proc * cols_per_proc);
        }
        return true;
    }

    bool TestMPITaskParallel::validation() {
        if (world.rank() == 0) {
            return taskData->inputs_count[0] == taskData->inputs_count[1];
        }
        return true;
    }

    bool TestMPITaskParallel::run() {
        local_matrix_c = strassen(local_matrix_a, local_matrix_b, rows_per_proc);
        return true;
    }

    bool TestMPITaskParallel::post_processing() {
        size_t total_size = rows_per_proc * cols_per_proc * world.size();
        std::vector<double> global_matrix_c(total_size, 0);
        
        boost::mpi::gather(world, local_matrix_c.data(), local_matrix_c.size(), global_matrix_c.data(), 0);

        if (world.rank() == 0) {
            double* output = reinterpret_cast<double*>(taskData->outputs[0]);
            std::copy(global_matrix_c.begin(), global_matrix_c.end(), output);
        }
        return true;
    }


    std::vector<double> TestMPITaskParallel::strassen(const std::vector<double>& A, const std::vector<double>& B, size_t n) {
        if (n <= 2) {
            std::vector<double> C(n * n, 0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        C[i * n + j] += A[i * n + k] * B[k * n + j];
                    }
                }
            }
            return C;
        }

        size_t half = n / 2;
        std::vector<double> a11(half * half), a12(half * half), a21(half * half), a22(half * half);
        std::vector<double> b11(half * half), b12(half * half), b21(half * half), b22(half * half);

        split_matrix(A, n, a11, a12, a21, a22);
        split_matrix(B, n, b11, b12, b21, b22);

        // 7 рекурсивных умножений по алгоритму Штрассена
        auto m1 = strassen(a11, b11, half);  // M1 = (A11 + A22)(B11 + B22)
        auto m2 = strassen(a21, b11, half);  // M2 = (A21 + A22)B11
        auto m3 = strassen(a11, b12, half);  // M3 = A11(B12 - B22)
        auto m4 = strassen(a22, b21, half);  // M4 = A22(B21 - B11)
        auto m5 = strassen(a11, b22, half);  // M5 = (A11 + A12)B22
        auto m6 = strassen(a21, b22, half);  // M6 = (A21 - A11)(B11 + B12)
        auto m7 = strassen(a12, b21, half);  // M7 = (A12 - A22)(B21 + B22)

        // Собираем результаты
        std::vector<double> c11 = combine_matrix(m1, m4, m5, m7, half);
        std::vector<double> c12 = combine_matrix(m3, m5, m7, m6, half);
        std::vector<double> c21 = combine_matrix(m2, m4, m6, m7, half);
        std::vector<double> c22 = combine_matrix(m1, m3, m2, m6, half);

        return combine_matrix(c11, c12, c21, c22, n);
    }


    void TestMPITaskParallel::split_matrix(const std::vector<double>& matrix, size_t n,
                                           std::vector<double>& a11, std::vector<double>& a12,
                                           std::vector<double>& a21, std::vector<double>& a22) {
        size_t half = n / 2;
        a11.resize(half * half);
        a12.resize(half * half);
        a21.resize(half * half);
        a22.resize(half * half);

        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
                a11[i * half + j] = matrix[i * n + j];
                a12[i * half + j] = matrix[i * n + j + half];
                a21[i * half + j] = matrix[(i + half) * n + j];
                a22[i * half + j] = matrix[(i + half) * n + j + half];
            }
        }
    }

    std::vector<double> TestMPITaskParallel::combine_matrix(const std::vector<double>& c11, const std::vector<double>& c12,
                                                            const std::vector<double>& c21, const std::vector<double>& c22, size_t n) {
        std::vector<double> matrix(n * n);
        size_t half = n / 2;

        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
                matrix[i * n + j] = c11[i * half + j];
                matrix[i * n + j + half] = c12[i * half + j];
                matrix[(i + half) * n + j] = c21[i * half + j];
                matrix[(i + half) * n + j + half] = c22[i * half + j];
            }
        }
        return matrix;
    }

}  // namespace nasedkin_e_strassen_algorithm
