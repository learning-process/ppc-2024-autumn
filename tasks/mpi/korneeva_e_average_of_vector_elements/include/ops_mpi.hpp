#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstring>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_average_of_vector_elements_mpi {

/////////// Sequential ///////////
template <class iotype>
class vector_average_sequential : public ppc::core::Task {
 public:
  explicit vector_average_sequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<iotype> input_data;  // Vector containing input data
  double average_result;           // Result of the average calculation
};
/////////////////////////////////////

/////////// MPI_Allreduce ///////////
template <class iotype>
class vector_average_MPI_AllReduce : public ppc::core::Task {
 public:
  explicit vector_average_MPI_AllReduce(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<iotype> input_data;     // Vector containing the input data (used only by the root process)
  std::vector<iotype> local_data;     // Vector holding the local data assigned to this process
  int input_size;                     // Total number of elements in the input vector
  int local_size;                     // Number of elements assigned to the local process
  double average_result;              // Result of the average calculation
  boost::mpi::communicator mpi_comm;  // MPI communicator
};
/////////////////////////////////////

/////////// my_Allreduce ///////////
template <class iotype>
class vector_average_my_AllReduce : public ppc::core::Task {
 public:
  explicit vector_average_my_AllReduce(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  // Custom implementation of the Allreduce operation.
  // Parameters:
  // - comm: MPI communicator.
  // - in_values: Input array of values.
  // - out_values: Output array of reduced values.
  // - n: Number of elements.
  // - op: Reduction operation (e.g., sum, max).
  template <typename T, typename Op>
  void my_AllReduce(const boost::mpi::communicator& comm, const T* in_values, T* out_values, int n, Op op);

 private:
  std::vector<iotype> input_data;
  std::vector<iotype> local_data;
  int input_size;
  int local_size;
  double average_result;
  boost::mpi::communicator mpi_comm;
};
/////////////////////////////////////

////////// class vector_average_sequential //////////
template <class iotype>
bool vector_average_sequential<iotype>::pre_processing() {
  internal_order_test();

  int input_size = taskData->inputs_count[0];
  const auto* source_ptr = reinterpret_cast<const iotype*>(taskData->inputs[0]);
  input_data.assign(source_ptr, source_ptr + input_size);

  return true;
}

template <class iotype>
bool vector_average_sequential<iotype>::validation() {
  internal_order_test();

  bool valid_output = (taskData->outputs_count[0] == 1);
  bool valid_inputs = (taskData->inputs_count.size() == 1) && (taskData->inputs_count[0] >= 0);

  return valid_output && valid_inputs;
}

template <class iotype>
bool vector_average_sequential<iotype>::run() {
  internal_order_test();

  // If the input vector is empty, set the result to 0.0 and return true.
  if (input_data.empty()) {
    average_result = 0.0;
    return true;
  }

  double sum = 0.0;
  for (const auto& value : input_data) {
    sum += static_cast<double>(value);  // Cast values to double for accurate summation
  }
  average_result = sum / static_cast<double>(input_data.size());

  return true;
}

template <class iotype>
bool vector_average_sequential<iotype>::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = average_result;
  return true;
}
////////////////////////////////////////////////////////

////////// class vector_average_MPI_AllReduce //////////
template <class iotype>
bool vector_average_MPI_AllReduce<iotype>::pre_processing() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    input_size = taskData->inputs_count[0];
    const auto* source_ptr = reinterpret_cast<const iotype*>(taskData->inputs[0]);
    input_data.assign(source_ptr, source_ptr + input_size);
  }
  return true;
}

template <class iotype>
bool vector_average_MPI_AllReduce<iotype>::validation() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    bool valid_output = (taskData->outputs_count[0] == 1);
    bool valid_inputs = (taskData->inputs_count.size() == 1) && (taskData->inputs_count[0] >= 0);

    return valid_output && valid_inputs;
  }
  return true;
}

template <class iotype>
bool vector_average_MPI_AllReduce<iotype>::run() {
  internal_order_test();
  int process_rank = mpi_comm.rank();
  int total_processes = mpi_comm.size();

  // Step 1: Broadcast the input vector size to all processes.
  boost::mpi::broadcast(mpi_comm, input_size, 0);

  // Handle the edge case where the input vector is empty.
  if (input_size <= 0) {
    if (process_rank == 0) {
      average_result = 0.0;
    }
    return true;
  }

  // Step 2: Calculate the local size of data for each process.
  local_size = input_size / total_processes + static_cast<int>(process_rank < input_size % total_processes);

  // Step 3: Scatter the input data among processes.
  local_data.resize(local_size);
  std::vector<int> send_counts(total_processes, 0);
  std::vector<int> send_offsets(total_processes, 0);

  // Only the root process calculates send_counts and send_offsets
  if (process_rank == 0) {
    for (int i = 0; i < total_processes; ++i) {
      send_counts[i] = input_size / total_processes + static_cast<int>(i < input_size % total_processes);
      if (i > 0) {
        send_offsets[i] = send_offsets[i - 1] + send_counts[i - 1];
      }
    }
  }
  boost::mpi::scatterv(mpi_comm, input_data, send_counts, send_offsets, local_data.data(), local_size, 0);

  // Step 4: Calculate the local sum of the data on each process
  double local_sum = 0.0;
  for (int i = 0; i < local_size; ++i) {
    local_sum += static_cast<double>(local_data[i]);
  }

  // Step 5: Perform an allreduce operation to collect and sum up the local sums from all processes
  double total_sum = 0.0;
  boost::mpi::all_reduce(mpi_comm, local_sum, total_sum, std::plus<>());

  // Step 6: Compute the final average (only the root process needs this).
  if (process_rank == 0) {
    average_result = total_sum / static_cast<double>(input_size);
  }
  return true;
}

template <class iotype>
bool vector_average_MPI_AllReduce<iotype>::post_processing() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = average_result;
  }
  return true;
}
////////////////////////////////////////////////////////

////////// class vector_average_my_AllReduce //////////
template <class iotype>
template <typename T, typename Op>
void vector_average_my_AllReduce<iotype>::my_AllReduce(const boost::mpi::communicator& comm, const T* in_values,
                                                       T* out_values, int n, Op op) {
  int rank = comm.rank();
  int size = comm.size();

  // Step 1: Initialize local values from the input array (in_values).
  std::vector<T> local_values(in_values, in_values + n);
  std::vector<T> buffer(n);

  // Step 2: Perform the reduction upwards in the binary tree.
  for (int child = 2 * rank + 1; child <= 2 * rank + 2; ++child) {
    if (child < size) {  // If the child exists within the process range.
      comm.recv(child, 0, buffer.data(), n);
      for (int i = 0; i < n; ++i) {
        local_values[i] = op(local_values[i], buffer[i]);
      }
    }
  }

  // Step 3: If this is not the root process, send the result to its parent.
  if (rank != 0) {
    int parent = (rank - 1) / 2;
    comm.send(parent, 0, local_values.data(), n);
    comm.recv(parent, 0, local_values.data(), n);
  }

  // Step 4: Perform the downward broadcast after the reduction is complete.
  for (int child = 2 * rank + 1; child <= 2 * rank + 2; ++child) {
    if (child < size) {
      comm.send(child, 0, local_values.data(), n);
    }
  }

  // Step 5: Copy the final result into the output array (out_values).
  std::copy(local_values.begin(), local_values.end(), out_values);
}

template <class iotype>
bool vector_average_my_AllReduce<iotype>::pre_processing() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    input_size = taskData->inputs_count[0];
    const auto* source_ptr = reinterpret_cast<const iotype*>(taskData->inputs[0]);
    input_data.assign(source_ptr, source_ptr + input_size);
  }
  return true;
}

template <class iotype>
bool vector_average_my_AllReduce<iotype>::validation() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    bool valid_output = (taskData->outputs_count[0] == 1);
    bool valid_inputs = (taskData->inputs_count.size() == 1) && (taskData->inputs_count[0] >= 0);

    return valid_output && valid_inputs;
  }
  return true;
}

template <class iotype>
bool vector_average_my_AllReduce<iotype>::run() {
  internal_order_test();
  int process_rank = mpi_comm.rank();
  int total_processes = mpi_comm.size();

  boost::mpi::broadcast(mpi_comm, input_size, 0);

  if (input_size <= 0) {
    if (process_rank == 0) {
      average_result = 0.0;
    }
    return true;
  }

  local_size = input_size / total_processes + static_cast<int>(process_rank < input_size % total_processes);

  local_data.resize(local_size);
  std::vector<int> send_counts(total_processes, 0);
  std::vector<int> send_offsets(total_processes, 0);

  if (process_rank == 0) {
    for (int i = 0; i < total_processes; ++i) {
      send_counts[i] = input_size / total_processes + static_cast<int>(i < input_size % total_processes);
      if (i > 0) {
        send_offsets[i] = send_offsets[i - 1] + send_counts[i - 1];
      }
    }
  }

  boost::mpi::scatterv(mpi_comm, input_data, send_counts, send_offsets, local_data.data(), local_size, 0);

  double local_sum = 0.0;
  for (int i = 0; i < local_size; ++i) {
    local_sum += static_cast<double>(local_data[i]);
  }

  double total_sum = 0.0;
  my_AllReduce(mpi_comm, &local_sum, &total_sum, 1, std::plus<>());

  if (process_rank == 0) {
    average_result = total_sum / static_cast<double>(input_size);
  }

  return true;
}

template <class iotype>
bool vector_average_my_AllReduce<iotype>::post_processing() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = average_result;
  }
  return true;
}
}  // namespace korneeva_e_average_of_vector_elements_mpi