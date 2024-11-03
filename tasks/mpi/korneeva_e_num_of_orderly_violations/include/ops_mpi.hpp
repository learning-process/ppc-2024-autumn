#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <cstring>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_num_of_orderly_violations_mpi {

template <class iotype, class cntype>
class num_of_orderly_violations : public ppc::core::Task {
 public:
   explicit num_of_orderly_violations(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}

   bool pre_processing() override;
   bool validation() override;
   bool run() override;
   bool post_processing() override;

   cntype count_orderly_violations(std::vector<iotype> vec);

 private:
   std::vector<iotype> input_data_;    // Local copy of data for processing
   cntype violation_count_;            // Variable to store count of violations
   boost::mpi::communicator mpi_comm;  // MPI communicator for parallel processing
   
   size_t local_vector_size_;          // Size of the local data vector
   std::vector<iotype> received_data_; // Buffer for data received from other processes
};

}  // namespace korneeva_e_num_of_orderly_violations_mpi
