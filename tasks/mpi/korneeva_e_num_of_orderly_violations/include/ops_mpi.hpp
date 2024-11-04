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

  size_t input_size;
  size_t local_vector_size_;           // Size of the local data vector
  std::vector<iotype> received_data_;  // Buffer for data received from other processes

  std::vector<int> send_sizes;
};

template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::pre_processing() {
  internal_order_test();
  int process_rank = mpi_comm.rank();
  input_size = 0;

  if (process_rank == 0) {
    input_size = taskData->inputs_count[0];

    if (input_size <= 1) {
      local_vector_size_ = 0;
      received_data_.clear();
      violation_count_ = 0;
      boost::mpi::broadcast(mpi_comm, local_vector_size_, 0);
      return true;
    }

    input_data_.resize(input_size);
    const iotype* source_ptr = reinterpret_cast<const iotype*>(taskData->inputs[0]);
    std::copy(source_ptr, source_ptr + input_size, input_data_.begin());

    violation_count_ = 0;
  }

  boost::mpi::broadcast(mpi_comm, input_size, 0);

  int total_processes = mpi_comm.size();
  int chunk_size = input_size / total_processes;
  int remainder = input_size % total_processes;

  send_sizes.resize(total_processes, 0);
  for (int i = 0; i < total_processes; ++i) {
    send_sizes[i] = chunk_size + (i < remainder ? 1 : 0);
  }

  local_vector_size_ = send_sizes[process_rank];

  if (local_vector_size_ == 0) {
    return true;
  }

  received_data_.resize(local_vector_size_);

  std::vector<int> offsets(total_processes, 0);
  for (int i = 1; i < total_processes; ++i) {
    offsets[i] = offsets[i - 1] + send_sizes[i - 1];
  }

  boost::mpi::scatterv(mpi_comm, input_data_, send_sizes, offsets, received_data_.data(), local_vector_size_, 0);
  return true;
}

template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::validation() {
  internal_order_test();  // Validate internal order

  // Process 0 checks the validity of input and output counts
  if (mpi_comm.rank() == 0) {
    bool valid_output = (taskData->outputs_count[0] == 1);
    bool valid_inputs = (taskData->inputs_count.size() == 1) && (taskData->inputs_count[0] >= 0);

    return valid_output && valid_inputs;  // Return true if both checks pass
  }
  return true;  // Other processes do not validate
}

template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::run() {
  internal_order_test();

  int rank = mpi_comm.rank();
  int size = mpi_comm.size();

  cntype local_violations = 0;

  if (local_vector_size_ > 1) {
    for (size_t index = 0; index < local_vector_size_ - 1; ++index) {
      if (received_data_[index + 1] < received_data_[index]) {
        local_violations++;
      }
    }
  }

  if (local_vector_size_ > 0) {
    iotype left_boundary, right_boundary;
    bool is_last_active_process = (rank == size - 1 || send_sizes[rank + 1] == 0);

    if (!is_last_active_process) {
      mpi_comm.recv(rank + 1, 0, right_boundary);
      if (received_data_[local_vector_size_ - 1] > right_boundary) {
        local_violations++;
      }
    }

    if (rank > 0) {
      left_boundary = received_data_[0];
      mpi_comm.send(rank - 1, 0, left_boundary);
    }
  }
  boost::mpi::reduce(mpi_comm, local_violations, violation_count_, std::plus<cntype>(), 0);
  return true;
}

template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::post_processing() {
  internal_order_test();  // Validate internal order

  // Process 0 writes the total violation count to output
  if (mpi_comm.rank() == 0) {
    auto output_ptr = reinterpret_cast<cntype*>(taskData->outputs[0]);
    output_ptr[0] = violation_count_;
  }
  return true;
}

template <typename iotype, typename cntype>
cntype num_of_orderly_violations<iotype, cntype>::count_orderly_violations(std::vector<iotype> vector_data) {
  cntype violation_count = 0;  // Initialize violation count

  // Return zero if the input vector is empty
  if (vector_data.empty()) {
    return violation_count;
  }

  // Count violations in the provided vector
  for (size_t index = 0; index < vector_data.size() - 1; ++index) {
    if (vector_data[index + 1] < vector_data[index]) {
      violation_count++;
    }
  }
  return violation_count;  // Return the total violation count
}

}  // namespace korneeva_e_num_of_orderly_violations_mpi
