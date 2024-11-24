#include "seq/nasedkin_e_seidels_iterate_methods/include/ops_seq.hpp"
#include <cmath>
#include <iostream>

namespace nasedkin_e_seidels_iterate_methods_seq {

    bool SeidelIterateMethodsSeq::pre_processing() {
        if (!validation()) {
            return false;
        }

        epsilon = 1e-6;
        max_iterations = 1000;

        A.resize(n, std::vector<double>(n, 0.0));
        b.resize(n, 0.0);
        x.resize(n, 0.0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = (i == j) ? 2.0 : 1.0;
            }
            b[i] = n + 1;
        }

        return true;
    }

    bool SeidelIterateMethodsSeq::validation() {
        if (taskData->inputs_count.empty()) {
            return false;
        }

        n = taskData->inputs_count[0];
        return n > 0;
    }

    bool SeidelIterateMethodsSeq::run() {
        std::vector<double> x_new(n, 0.0);
        int iteration = 0;

        while (iteration < max_iterations) {
            for (int i = 0; i < n; ++i) {
                x_new[i] = b[i];
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        x_new[i] -= A[i][j] * x[j];
                    }
                }
                x_new[i] /= A[i][i];
            }

            if (converge(x_new)) {
                break;
            }

            x = x_new;
            ++iteration;
        }

        return true;
    }

    bool SeidelIterateMethodsSeq::post_processing() {
        std::cout << "Solution: ";
        for (const auto& val : x) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        return true;
    }

    bool SeidelIterateMethodsSeq::converge(const std::vector<double>& x_new) {
        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        return std::sqrt(norm) < epsilon;
    }

}  // namespace nasedkin_e_seidels_iterate_methods_seq
