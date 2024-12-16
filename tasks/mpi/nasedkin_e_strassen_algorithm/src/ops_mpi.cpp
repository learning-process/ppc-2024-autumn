#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <iostream>

namespace nasedkin_e_strassen_algorithm_mpi {

    bool StrassenAlgorithmMPI::pre_processing() {
        if (!validation()) {
            return false;
        }

        C.resize(n, std::vector<double>(n, 0.0));

        return true;
    }

    bool StrassenAlgorithmMPI::validation() {
        if (taskData->inputs_count.empty()) {
            return false;
        }

        n = taskData->inputs_count[0];
        if (n <= 0 || (n & (n - 1)) != 0) {
            return false;
        }

        A.resize(n, std::vector<double>(n, 0.0));
        B.resize(n, std::vector<double>(n, 0.0));

        return true;
    }

    bool StrassenAlgorithmMPI::run() {
        int rank = world.rank();
        int size = world.size();

        std::vector<double> flat_A, flat_B, flat_C;

        if (rank == 0) {
            flatten_matrix(A, flat_A);
            flatten_matrix(B, flat_B);
        }

        boost::mpi::broadcast(world, flat_A, 0);
        boost::mpi::broadcast(world, flat_B, 0);

        std::vector<std::vector<double>> local_A, local_B;
        unflatten_matrix(flat_A, local_A, n);
        unflatten_matrix(flat_B, local_B, n);
        std::vector<std::vector<double>> local_C(n / size, std::vector<double>(n, 0.0));
        strassen_recursive(local_A, local_B, local_C, n / size);

        std::vector<double> flat_local_C;
        flatten_matrix(local_C, flat_local_C);
        boost::mpi::all_reduce(world, flat_local_C, flat_C, std::plus<>());
        unflatten_matrix(flat_C, C, n);

        return true;
    }

    bool StrassenAlgorithmMPI::post_processing() { return true; }

    void StrassenAlgorithmMPI::set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB) {
        A = matrixA;
        B = matrixB;
        n = static_cast<int>(matrixA.size());
    }

    void StrassenAlgorithmMPI::set_matrices(const std::vector<double>& flatA, const std::vector<double>& flatB, int size) {
        unflatten_matrix(flatA, A, size);
        unflatten_matrix(flatB, B, size);
        n = size;
    }

    void StrassenAlgorithmMPI::generate_random_matrix(int size, std::vector<std::vector<double>>& matrix) {
        matrix.resize(size, std::vector<double>(size, 0.0));
        std::srand(static_cast<unsigned>(std::time(nullptr)));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = static_cast<double>(std::rand() % 100);
            }
        }
    }

    void StrassenAlgorithmMPI::strassen_recursive(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        if (size <= 64) {
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    for (int k = 0; k < size; ++k) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return;
        }

        int new_size = size / 2;
        std::vector<std::vector<double>> a11(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> a12(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> a21(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> a22(new_size, std::vector<double>(new_size));

        std::vector<std::vector<double>> b11(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> b12(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> b21(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> b22(new_size, std::vector<double>(new_size));

        std::vector<std::vector<double>> c11(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> c12(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> c21(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> c22(new_size, std::vector<double>(new_size));

        split_matrix(A, a11, a12, a21, a22, new_size);
        split_matrix(B, b11, b12, b21, b22, new_size);

        std::vector<std::vector<double>> p1(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> p2(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> p3(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> p4(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> p5(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> p6(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> p7(new_size, std::vector<double>(new_size));

        std::vector<std::vector<double>> temp1(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> temp2(new_size, std::vector<double>(new_size));

        add_matrices(a11, a22, temp1, new_size);
        add_matrices(b11, b22, temp2, new_size);
        strassen_recursive(temp1, temp2, p1, new_size);

        add_matrices(a21, a22, temp1, new_size);
        strassen_recursive(temp1, b11, p2, new_size);

        subtract_matrices(b12, b22, temp1, new_size);
        strassen_recursive(a11, temp1, p3, new_size);

        subtract_matrices(b21, b11, temp1, new_size);
        strassen_recursive(a22, temp1, p4, new_size);

        add_matrices(a11, a12, temp1, new_size);
        strassen_recursive(temp1, b22, p5, new_size);

        subtract_matrices(a21, a11, temp1, new_size);
        add_matrices(b11, b12, temp2, new_size);
        strassen_recursive(temp1, temp2, p6, new_size);

        subtract_matrices(a12, a22, temp1, new_size);
        add_matrices(b21, b22, temp2, new_size);
        strassen_recursive(temp1, temp2, p7, new_size);

        add_matrices(p1, p4, temp1, new_size);
        subtract_matrices(temp1, p5, temp2, new_size);
        add_matrices(temp2, p7, c11, new_size);

        add_matrices(p3, p5, c12, new_size);

        add_matrices(p2, p4, c21, new_size);

        add_matrices(p1, p3, temp1, new_size);
        subtract_matrices(temp1, p2, temp2, new_size);
        add_matrices(temp2, p6, c22, new_size);

        combine_matrices(C, c11, c12, c21, c22, new_size);
    }

    void StrassenAlgorithmMPI::add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
    }

    void StrassenAlgorithmMPI::subtract_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
    }

    void StrassenAlgorithmMPI::split_matrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& top_left, std::vector<std::vector<double>>& top_right, std::vector<std::vector<double>>& bottom_left, std::vector<std::vector<double>>& bottom_right, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                top_left[i][j] = matrix[i][j];
                top_right[i][j] = matrix[i][j + size];
                bottom_left[i][j] = matrix[i + size][j];
                bottom_right[i][j] = matrix[i + size][j + size];
            }
        }
    }

    void StrassenAlgorithmMPI::combine_matrices(std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& top_left, const std::vector<std::vector<double>>& top_right, const std::vector<std::vector<double>>& bottom_left, const std::vector<std::vector<double>>& bottom_right, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = top_left[i][j];
                matrix[i][j + size] = top_right[i][j];
                matrix[i + size][j] = bottom_left[i][j];
                matrix[i + size][j + size] = bottom_right[i][j];
            }
        }
    }

    void StrassenAlgorithmMPI::flatten_matrix(const std::vector<std::vector<double>>& matrix, std::vector<double>& flat) {
        flat.clear();
        for (const auto& row : matrix) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }

    void StrassenAlgorithmMPI::unflatten_matrix(const std::vector<double>& flat, std::vector<std::vector<double>>& matrix, int n) {
        matrix.clear();
        matrix.resize(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix[i][j] = flat[i * n + j];
            }
        }
    }

}  // namespace nasedkin_e_strassen_algorithm_mpi