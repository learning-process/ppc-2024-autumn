#include "mpi/korneeva_e_num_of_orderly_violations/include/ops_mpi.hpp"

namespace korneeva_e_num_of_orderly_violations_mpi {

template <class iotype, class cntype>
bool num_of_orderly_violations<iotype, cntype>::pre_processing() {
  internal_order_test();                  // Validate internal order
  int process_rank = mpi_comm.rank();     // Get the rank of the current process
  int total_processes = mpi_comm.size();  // Get the total number of processes
  int input_size = 0;

  // Process 0 initializes input data
  if (process_rank == 0) {
    input_size = taskData->inputs_count[0];
    input_data_ = std::vector<iotype>(input_size);
    void* source_ptr = taskData->inputs[0];
    void* dest_ptr = input_data_.data();
    std::memcpy(dest_ptr, source_ptr, sizeof(iotype) * input_size);
    violation_count_ = 0;  // Initialize violation count
  }

  // Broadcast the input size to all processes
  boost::mpi::broadcast(mpi_comm, input_size, 0);

  // Handle edge cases where input size is 0 or 1
  if (input_size == 0 || input_size == 1) {
    local_vector_size_ = 0;
    received_data_.clear();
    return true;
  }

  // Calculate the chunk size for each process
  int active_processes = std::min(input_size, total_processes);
  int chunk_size = input_size / active_processes;
  int remainder = input_size % active_processes;

  // Prepare send sizes for each process
  std::vector<int> send_sizes(total_processes, 0);  // By default, no data is assigned to any process
  for (int i = 0; i < active_processes; ++i) {
    send_sizes[i] = chunk_size + (i < remainder ? 1 : 0);  // First `remainder` processes receive one extra element
  }

  local_vector_size_ = send_sizes[process_rank];  // Local data size for this process
  if (local_vector_size_ <= 0) {
    return true;  // If no data is assigned to this process, exit early
  }

  // If the process has data to process, allocate a buffer
  if (local_vector_size_ > 0) {
    received_data_.resize(local_vector_size_);
  }

  // Perform data scatter across processes
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
  internal_order_test();  // Validate internal order

  cntype local_violations = 0;  // Initialize local violation count

  // If there are no local data, exit early
  if (local_vector_size_ == 0) {
    return true;
  }

  // Prepare boundary values for violation checking
  iotype left_boundary;
  iotype right_boundary = received_data_.back();  // Last element of the local data

  int rank = mpi_comm.rank();
  int size = mpi_comm.size();

  // Send right boundary to the next process
  if (rank < size - 1) {
    mpi_comm.send(rank + 1, 0, right_boundary);
  }

  // Receive left boundary from the previous process
  if (rank > 0) {
    mpi_comm.recv(rank - 1, 0, left_boundary);

    // Check for violation with the left boundary
    if (left_boundary > received_data_[0]) {
      local_violations++;
    }
  }

  // Count violations within the local data
  for (size_t index = 0; index < local_vector_size_ - 1; ++index) {
    if (received_data_[index + 1] < received_data_[index]) {
      local_violations++;
    }
  }

  // Reduce the local violation counts to the root process
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
