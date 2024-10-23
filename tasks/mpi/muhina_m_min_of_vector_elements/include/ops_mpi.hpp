// Copyright 2023 Nesterov Alexander
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

namespace muhina_m_min_of_vector_elements_mpi {

    std::vector<int> GetRandomVector(int sz);
    int vectorMin(std::vector<int, std::allocator<int>> v);


    class MinOfVectorMPISequential : public ppc::core::Task {
    public:
      explicit MinOfVectorMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        std::vector<int> input_;
        int res{};
        std::string ops;
    };


    class MinOfVectorMPIParallel : public ppc::core::Task {
    public:
        explicit MinOfVectorMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        std::vector<int> input_, local_input_;
        int res{};
        std::string ops;
        boost::mpi::communicator world;
    };

}  // namespace muhina_m_min_of_vector_elements_mpi