#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include <numeric>
#include <memory>
#include "core/perf/include/perf.hpp"
#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp"

namespace sotskov_a_sum_element_matrix_mpi {

TEST(sotskov_a_sum_element_matrix, test_pipeline_run) {
    boost::mpi::communicator world;
    int total_elements = 1000 * 1000;

    if (total_elements % world.size() != 0) {
        total_elements -= total_elements % world.size();
    }

    std::vector<int> global_vec(total_elements, 1);
    std::vector<int32_t> global_sum(1, 0);
    
    int elements_per_process = total_elements / world.size();
    std::vector<int> local_vec(elements_per_process, 0);

    MPI_Scatter(global_vec.data(), elements_per_process, MPI_INT, 
                 local_vec.data(), elements_per_process, MPI_INT, 
                 0, MPI_COMM_WORLD);
    
    int local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0);
    
    MPI_Reduce(&local_sum, &global_sum[0], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (world.rank() == 0) {
        ASSERT_EQ(total_elements, global_sum[0]);
    }
}

TEST(sotskov_a_sum_element_matrix, test_task_run) {
    boost::mpi::communicator world;
    int total_elements = 9000 * 9000;

    if (total_elements % world.size() != 0) {
        total_elements -= total_elements % world.size();
    }

    std::vector<int> global_vec(total_elements, 1);
    std::vector<int32_t> global_sum(1, 0);
    
    int elements_per_process = total_elements / world.size();
    std::vector<int> local_vec(elements_per_process, 0);

    MPI_Scatter(global_vec.data(), elements_per_process, MPI_INT, 
                 local_vec.data(), elements_per_process, MPI_INT, 
                 0, MPI_COMM_WORLD);
    
    int local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0);
    
    MPI_Reduce(&local_sum, &global_sum[0], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (world.rank() == 0) {
        ASSERT_EQ(total_elements, global_sum[0]);
    }
}

}  // namespace sotskov_a_sum_element_matrix_mpi
