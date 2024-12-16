#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

    class StrassenAlgorithmMPI : public ppc::core::Task {
    public:
        explicit StrassenAlgorithmMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

        void set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB);
        static void generate_random_matrix(int size, std::vector<std::vector<double>>& matrix);
        const std::vector<std::vector<double>>& get_result() const { return C; }

    private:
        boost::mpi::communicator world;
        std::vector<std::vector<double>> A;
        std::vector<std::vector<double>> B;
        std::vector<std::vector<double>> C;
        int n;

        void strassen_recursive(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        void add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        void subtract_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        void split_matrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& top_left, std::vector<std::vector<double>>& top_right, std::vector<std::vector<double>>& bottom_left, std::vector<std::vector<double>>& bottom_right, int size);
        void combine_matrices(std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& top_left, const std::vector<std::vector<double>>& top_right, const std::vector<std::vector<double>>& bottom_left, const std::vector<std::vector<double>>& bottom_right, int size);
    };

}  // namespace nasedkin_e_strassen_algorithm_mpi