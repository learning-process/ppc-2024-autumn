#pragma once

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>
#include <boost/mpi/communicator.hpp>
#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

    class StrassenAlgorithmSEQ : public ppc::core::Task {
    public:
        explicit StrassenAlgorithmSEQ(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        static std::vector<double> strassen_multiply_seq(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);

        std::vector<double> inputMatrixA;
        std::vector<double> inputMatrixB;
        std::vector<double> outputMatrix;
        size_t matrixSize;
    };

    class StrassenAlgorithmMPI : public ppc::core::Task {
    public:
        explicit StrassenAlgorithmMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        static std::vector<double> strassen_multiply(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);

        boost::mpi::communicator world;

        std::vector<double> inputMatrixA;
        std::vector<double> inputMatrixB;
        std::vector<double> outputMatrix;
        size_t matrixSize;
    };

    std::vector<double> strassen_recursive(const std::vector<double>& matrixA,
                                                  const std::vector<double>& matrixB, size_t size);
    std::vector<double> matrix_add(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);
    std::vector<double> matrix_subtract(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);
}  // namespace nasedkin_e_strassen_algorithm