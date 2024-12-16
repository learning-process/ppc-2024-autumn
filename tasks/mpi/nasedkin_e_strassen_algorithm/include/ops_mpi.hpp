#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

    class StrassenAlgorithmMPI : public ppc::core::Task {
    public:
        explicit StrassenAlgorithmMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

        void set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB);
        static void generate_random_matrix(int size, std::vector<std::vector<double>>& matrix);
        const std::vector<std::vector<double>>& get_result() const { return result; }

    private:
        boost::mpi::communicator world;
        std::vector<std::vector<double>> A;
        std::vector<std::vector<double>> B;
        std::vector<std::vector<double>> result;
        int n;

        void strassen_multiply(const std::vector<std::vector<double>>& local_A, const std::vector<std::vector<double>>& local_B, std::vector<std::vector<double>>& local_result, int size);
        static void add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        static void subtract_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        static void split_matrix(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A11, std::vector<std::vector<double>>& A12, std::vector<std::vector<double>>& A21, std::vector<std::vector<double>>& A22, int size);
        static void join_matrices(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22, std::vector<std::vector<double>>& C, int size);
        void distribute_matrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& distributed_matrix);
        void gather_result(const std::vector<std::vector<double>>& local_result, std::vector<std::vector<double>>& result);
        static void flatten_matrix(const std::vector<std::vector<double>>& matrix, std::vector<double>& flat_matrix);
        static void unflatten_matrix(const std::vector<double>& flat_matrix, std::vector<std::vector<double>>& matrix, int rows, int cols);
    };

}  // namespace nasedkin_e_strassen_algorithm