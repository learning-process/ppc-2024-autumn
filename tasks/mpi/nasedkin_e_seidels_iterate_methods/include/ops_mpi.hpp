#pragma once

#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>
#include "core/task/include/task.hpp"

namespace nasedkin_e_seidels_iterate_methods_mpi {

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

    private:
        int _rows{}, _columns{};
        std::vector<double> _coefs;
        std::vector<double> _b;
        std::vector<double> _x;
        boost::mpi::communicator world;
        std::vector<double> SeidelIterateMethod(const std::vector<double>& matrix, int rows, int cols, const std::vector<double>& vec);
    };

}  // namespace nasedkin_e_seidels_iterate_methods_mpi