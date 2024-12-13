#pragma once

#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>
#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

    std::vector<int> getRandomMatrix(int size);
    std::vector<int> matrixMultiply(const std::vector<int>& A, const std::vector<int>& B, int size);

    class StrassenMPITaskParallel : public ppc::core::Task {
    public:
        explicit StrassenMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
                : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        std::vector<int> A, B, C;
        int size;
        boost::mpi::communicator world;
    };

}  // namespace nasedkin_e_strassen_algorithm