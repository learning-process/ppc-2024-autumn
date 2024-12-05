#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>
#include <cmath>

namespace nasedkin_e_seidels_iterate_methods_mpi {

    bool TestMPITaskSequential::pre_processing() {
        internal_order_test();
        coefs = std::vector<double>(taskData->inputs_count[0]);
        auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
        for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
            coefs[i] = ptr[i];
        }
        b = std::vector<double>(taskData->inputs_count[1]);
        auto* ptr1 = reinterpret_cast<double*>(taskData->inputs[1]);
        for (unsigned int i = 0; i < taskData->inputs_count[1]; i++) {
            b[i] = ptr1[i];
        }
        columns = taskData->inputs_count[2];
        rows = taskData->inputs_count[3];
        return true;
    }

    bool TestMPITaskSequential::validation() {
        internal_order_test();
        if (taskData->inputs.size() == 2 && taskData->outputs.size() == 1 && taskData->inputs_count.size() == 4 &&
            taskData->outputs_count.size() == 1) {
            return (taskData->inputs_count[3] == taskData->inputs_count[2] &&
                    taskData->inputs_count[2] == taskData->outputs_count[0]) &&
                   taskData->inputs.size() == 2 && taskData->outputs.size() == 1;
        }
        return false;
    }

    bool TestMPITaskSequential::run() {
        internal_order_test();
        x = SeidelIterateMethod(coefs, rows, columns, b);
        return true;
    }

    bool TestMPITaskSequential::post_processing() {
        internal_order_test();
        for (int i = 0; i < columns; i++) {
            reinterpret_cast<double*>(taskData->outputs[0])[i] = x[i];
        }
        return true;
    }

    std::vector<double> TestMPITaskParallel::SeidelIterateMethod(const std::vector<double>& matrix, int rows, int cols, const std::vector<double>& vec) {
        std::vector<double> x(rows, 0);
        std::vector<double> x_new(rows, 0);
        double eps = 1e-9;
        bool converged = false;

        while (!converged) {
            converged = true;
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int j = 0; j < cols; j++) {
                    if (i != j) {
                        sum += matrix[i * cols + j] * x[j];
                    }
                }
                x_new[i] = (vec[i] - sum) / matrix[i * cols + i];
                if (std::abs(x_new[i] - x[i]) > eps) {
                    converged = false;
                }
            }
            x = x_new;
        }
        return x;
    }

    bool TestMPITaskParallel::pre_processing() {
        internal_order_test();
        _coefs = std::vector<double>(taskData->inputs_count[0]);
        auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
        for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
            _coefs[i] = ptr[i];
        }
        _b = std::vector<double>(taskData->inputs_count[1]);
        auto* ptr1 = reinterpret_cast<double*>(taskData->inputs[1]);
        for (unsigned int i = 0; i < taskData->inputs_count[1]; i++) {
            _b[i] = ptr1[i];
        }
        _columns = taskData->inputs_count[2];
        _rows = taskData->inputs_count[3];
        return true;
    }

    bool TestMPITaskParallel::validation() {
        internal_order_test();
        if (taskData->inputs.size() == 2 && taskData->outputs.size() == 1 && taskData->inputs_count.size() == 4 &&
            taskData->outputs_count.size() == 1) {
            return (taskData->inputs_count[3] == taskData->inputs_count[2] &&
                    taskData->inputs_count[2] == taskData->outputs_count[0]) &&
                   taskData->inputs.size() == 2 && taskData->outputs.size() == 1;
        }
        return false;
    }

    bool TestMPITaskParallel::run() {
        internal_order_test();
        _x = SeidelIterateMethod(_coefs, _rows, _columns, _b);
        return true;
    }

    bool TestMPITaskParallel::post_processing() {
        internal_order_test();
        for (int i = 0; i < _columns; i++) {
            reinterpret_cast<double*>(taskData->outputs[0])[i] = _x[i];
        }
        return true;
    }

}  // namespace nasedkin_e_seidels_iterate_methods_mpi