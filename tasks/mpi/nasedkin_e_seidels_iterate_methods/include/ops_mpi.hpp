#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_seidels_iterate_methods_mpi {

    std::vector<double> generateDenseMatrix(int n, int a);
    std::vector<double> generateElementaryMatrix(int rows, int columns);
    template <typename T>
    std::vector<T> getRandomVector(int sz);

    class TestMPITaskSequential : public ppc::core::Task {
    public:
        explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        int rows{}, columns{};
        std::vector<double> coefs;
        std::vector<double> b;
        std::vector<double> x;
    };

    class TestMPITaskParallel : public ppc::core::Task {
    public:
        explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

        static std::vector<double> seidelMethod(const std::vector<double>& A, const std::vector<double>& b, int n, double eps);
        static bool hasZeroDiagonal(const std::vector<double>& matrix, int n);

    private:
        int _rows{}, _columns{};
        std::vector<double> _coefs;
        std::vector<double> _b;
        std::vector<double> _x;
        boost::mpi::communicator world;
    };

}  // namespace nasedkin_e_seidels_iterate_methods_mpi