#include "mpi/prokhorov_n_producer_customer/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <numeric>
#include <vector>

namespace prokhorov_n_producer_customer_mpi {

std::vector<int> buffer;
int buffer_capacity = 20;
int buffer_size = 0;
std::mutex buffer_mutex;
bool processing_complete = false;

std::vector<int> getPredefinedVector(int sz) { return std::vector<int>(sz, 1); }

bool prokhorov_n_producer_customer_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.size() < 3) {
    if (world.rank() == 0) {
    }
    return false;
  }

  unsigned int total_data_count = 0;

  if (world.rank() == 0) {
    if (producer_data.empty()) {
      producer_data = getPredefinedVector(50);
    }

    total_data_count = producer_data.size();
    if (total_data_count == 0) {
      return false;
    }
  }

  broadcast(world, total_data_count, 0);

  return true;
}

bool prokhorov_n_producer_customer_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return world.size() > 1;
  }
  return true;
}

bool prokhorov_n_producer_customer_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int data_size = 0;

  broadcast(world, data_size, 0);

  if (data_size <= 0) {
    if (world.rank() == 0) {
    }
    return false;
  }

  if (world.rank() == 0) {
    int chunk_size = data_size / (world.size() - 2);
    int remainder = data_size % (world.size() - 2);
    int offset = 0;

    for (int i = 1; i < world.size() - 1; ++i) {
      int current_chunk = chunk_size + (i == world.size() - 2 ? remainder : 0);
      if (current_chunk <= 0) {
        return false;
      }
      world.send(i, 0, producer_data.data() + offset, current_chunk);
      offset += current_chunk;
    }
  } else if (world.rank() < world.size() - 1) {
    int chunk_size = data_size / (world.size() - 2);
    if (world.rank() == world.size() - 2) {
      chunk_size += data_size % (world.size() - 2);
    }
    local_input_.resize(chunk_size);

    world.recv(0, 0, local_input_.data(), chunk_size);

    for (int val : local_input_) {
      {
        std::unique_lock<std::mutex> lock(buffer_mutex);
        while (buffer_size >= buffer_capacity && !processing_complete) {
          lock.unlock();
          MPI_Barrier(MPI_COMM_WORLD);
          lock.lock();
        }
        if (processing_complete) break;

        buffer.push_back(val);
        buffer_size++;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  } else {
    std::vector<int> consumer_results;

    while (consumer_results.size() < static_cast<size_t>(data_size)) {
      {
        std::unique_lock<std::mutex> lock(buffer_mutex);
        while (buffer_size == 0 && !processing_complete) {
          lock.unlock();
          MPI_Barrier(MPI_COMM_WORLD);
          lock.lock();
        }
        if (processing_complete) break;

        int data = buffer.back();
        buffer.pop_back();
        buffer_size--;

        consumer_results.push_back(data * 2);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    world.send(0, 0, consumer_results.data(), consumer_results.size());
  }

  if (world.rank() == 0) {
    processing_complete = true;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool prokhorov_n_producer_customer_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->outputs_count[0] != 1) {
      return false;
    }

    int total_result = 0;

    for (int i = 1; i < world.size(); ++i) {
      std::vector<int> local_results(buffer_capacity, 0);
      world.recv(i, 0, local_results.data(), buffer_capacity);
      total_result += std::accumulate(local_results.begin(), local_results.end(), 0);
    }

    reinterpret_cast<int *>(taskData->outputs[0])[0] = total_result;
  }

  return true;
}

}  // namespace prokhorov_n_producer_customer_mpi
