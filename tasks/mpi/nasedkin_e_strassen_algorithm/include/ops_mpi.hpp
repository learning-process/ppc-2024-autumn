#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <memory>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

    class TestMPITaskParallel {
    public:
        explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData);
        bool pre_processing();
        bool validation();
        bool run();
        bool post_processing();

    private:
        boost::mpi::communicator world;
        std::shared_ptr<ppc::core::TaskData> taskData;

        std::vector<double> local_matrix_a, local_matrix_b, local_matrix_c;
        size_t rows_per_proc, cols_per_proc;

        std::vector<double> strassen(const std::vector<double>& A, const std::vector<double>& B, size_t n);
        static void split_matrix(const std::vector<double>& matrix, size_t n,
                          std::vector<double>& a11, std::vector<double>& a12,
                          std::vector<double>& a21, std::vector<double>& a22);
        static std::vector<double> combine_matrix(const std::vector<double>& c11, const std::vector<double>& c12,
                                           const std::vector<double>& c21, const std::vector<double>& c22, size_t n);
    };

}  // namespace nasedkin_e_strassen_algorithm
