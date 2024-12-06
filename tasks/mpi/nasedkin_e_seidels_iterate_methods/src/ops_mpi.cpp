#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <boost/serialization/vector.hpp>

namespace nasedkin_e_seidels_iterate_methods_mpi {

    std::vector<double> generateDenseMatrix(int n, int a) {
        std::vector<double> dense;
        std::vector<double> ed(n * n);
        std::vector<double> res(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n + i; j++) {
                dense.push_back(a + j);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i < 2) {
                    ed[j * n + i] = 0;
                } else if (i == j) {
                    ed[j * n + i] = 1;
                } else {
                    ed[j * n + i] = 0;
                }
            }
        }
        for (int i = 0; i < n * n; i++) {
            res[i] = (dense[i] + ed[i]);
        }
        return res;
    }

    std::vector<double> generateElementaryMatrix(int rows, int columns) {
        std::vector<double> res;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (i == j) {
                    res.push_back(1);
                } else {
                    res.push_back(0);
                }
            }
        }
        return res;
    }

    template <typename T>
    std::vector<T> getRandomVector(int sz) {
        std::random_device dev;
        std::mt19937 gen(dev());
        std::vector<T> vec(sz);
        vec[0] = gen() % 100;
        for (int i = 1; i < sz; i++) {
            vec[i] = (gen() % 100) - 49;
        }
        return vec;
    }

    template std::vector<int> nasedkin_e_seidels_iterate_methods_mpi::getRandomVector(int sz);
    template std::vector<double> nasedkin_e_seidels_iterate_methods_mpi::getRandomVector(int sz);

    std::vector<double> nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::seidelMethod(
            const std::vector<double>& A, const std::vector<double>& b, int n, double eps) {
        std::vector<double> x(n, 0.0);
        std::vector<double> x_new(n, 0.0);
        double norm;
        do {
            norm = 0.0;
            for (int i = 0; i < n; ++i) {
                x_new[i] = b[i];
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        x_new[i] -= A[i * n + j] * x[j];
                    }
                }
                if (A[i * n + i] != 0.0) {
                    x_new[i] /= A[i * n + i];
                }
                norm += std::abs(x_new[i] - x[i]);
            }
            x = x_new;
        } while (norm > eps);
        return x;
    }


    bool nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::hasZeroDiagonal(const std::vector<double>& matrix, int n) {
        for (int i = 0; i < n; ++i) {
            if (matrix[i * n + i] == 0.0) {
                return true;
            }
        }
        return false;
    }

    bool nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::pre_processing() {
        internal_order_test();

        if (world.rank() == 0) {
            _rows = taskData->inputs_count[3];
            _columns = taskData->inputs_count[2];
        }

        broadcast(world, _rows, 0);
        broadcast(world, _columns, 0);

        if (world.rank() == 0) {
            _coefs = std::vector<double>(taskData->inputs_count[0]);
            auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
            for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
                _coefs[i] = tmp_ptr[i];
            }
            _b = std::vector<double>(taskData->inputs_count[1]);
            auto* ptr1 = reinterpret_cast<double*>(taskData->inputs[1]);
            for (unsigned int i = 0; i < taskData->inputs_count[1]; i++) {
                _b[i] = ptr1[i];
            }
        } else {
            _coefs.resize(_columns * _rows);
            _b.resize(_rows);
        }

        broadcast(world, _coefs, 0);
        broadcast(world, _b, 0);

        _x = std::vector<double>(_rows, 0);
        return true;
    }



    bool nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::validation() {
        internal_order_test();

        if (world.rank() == 0) {
            if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1 ||
                taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
                return false;
            }
            if (taskData->inputs_count[2] <= 0 || taskData->inputs_count[3] <= 0) {
                return false;
            }
            std::vector<double> tmp_coefs(taskData->inputs_count[0]);
            auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
            for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
                tmp_coefs[i] = tmp_ptr[i];
            }
            int n = taskData->inputs_count[3];
            bool hasZero = hasZeroDiagonal(tmp_coefs, n);
            broadcast(world, hasZero, 0);
            return !hasZero;
        }

        bool valid;
        broadcast(world, valid, 0);
        return valid;
    }




    bool nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::run() {
        internal_order_test();
        broadcast(world, _columns, 0);
        broadcast(world, _rows, 0);
        _x = nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::seidelMethod(_coefs, _b, _rows, 1e-6);
        return true;
    }

    bool nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel::post_processing() {
        internal_order_test();
        if (world.rank() == 0) {
            for (int i = 0; i < _columns; i++) {
                reinterpret_cast<double*>(taskData->outputs[0])[i] = _x[i];
            }
        }
        return true;
    }

}  // namespace nasedkin_e_seidels_iterate_methods_mpi