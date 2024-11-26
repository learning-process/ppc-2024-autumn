#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace guseynov_e_scatter{
    
    std::vector<int> getRandomVector(int sz);

    class TestMPITaskSequential : public ppc::core::Task{
        public:
        explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
            : Task(std::move(taskData_)){}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        std::vector<int> input_;
        int res_{};
    }

    class TestMPITaskParallel : ppc::core::Task{
        public:
         explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
            : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        std::vector<int> input_, local_input_;
        int res_{};
        boost::mpi::communicator world;
    }

    class MyScatterTestMPITaskParallel : ppc::core::Task{
        public:
         explicit MyScatterTestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
            : Task(std::move(taskData_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;
        template<typename T>
        static void my_scatter(const boost::mpi::communicator & comm, const std::vector<T> & in_values, T * out_values, int root);
        
    private:
        std::vector<int> input_, local_input_;
        int res_{};
        boost::mpi::communicator world;
    }
}