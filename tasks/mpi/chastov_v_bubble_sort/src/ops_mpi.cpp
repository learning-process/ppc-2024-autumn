#include "mpi/chastov_v_bubble_sort/include/ops_mpi.hpp"

#include <algorithm>

using namespace chastov_v_bubble_sort;

template <class T>
bool TestMPITaskParallel<T>::bubble_sort() {
  for (size_t i = 0; i < chunk_data.size() - 1; i++) {
    for (size_t j = 0; j < chunk_data.size() - i - 1; j++) {
      if (chunk_data[j] > chunk_data[j + 1]) {
        std::swap(chunk_data[j], chunk_data[j + 1]);
      }
    }
  }
  return true;
}

template <class T>
bool TestMPITaskParallel<T>::chunk_merge_sort(int neighbor_rank, std::vector<int>& chunk_sizes) {
  if (neighbor_rank >= 0 && neighbor_rank < world.size()) {
    std::vector<T> buffer;
    std::vector<T> merged_result;
    MPI_Request send_request;
    MPI_Request recv_request;

    int active_process = std::max(world.rank(), neighbor_rank);

    if (world.rank() == active_process) {
      buffer.resize(chunk_sizes[neighbor_rank]);
      MPI_Irecv(buffer.data(), chunk_sizes[neighbor_rank] * sizeof(T), MPI_BYTE, neighbor_rank, 0, MPI_COMM_WORLD,
                &recv_request);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      merged_result.clear();
      buffer.insert(buffer.end(), chunk_data.begin(), chunk_data.end());
      size_t left_idx = 0;
      size_t right_idx = chunk_sizes[neighbor_rank];
      while (right_idx < buffer.size() || left_idx < static_cast<size_t>(chunk_sizes[neighbor_rank])) {
        if ((left_idx < static_cast<size_t>(chunk_sizes[neighbor_rank]) && right_idx < buffer.size() &&
             buffer[left_idx] <= buffer[right_idx]) ||
            (left_idx < static_cast<size_t>(chunk_sizes[neighbor_rank]) && right_idx == buffer.size())) {
          merged_result.push_back(buffer[left_idx]);
          left_idx++;
        } else if ((left_idx < static_cast<size_t>(chunk_sizes[neighbor_rank]) && right_idx < buffer.size() &&
                    buffer[left_idx] >= buffer[right_idx]) ||
                   (left_idx == static_cast<size_t>(chunk_sizes[neighbor_rank]) && right_idx < buffer.size())) {
          merged_result.push_back(buffer[right_idx]);
          right_idx++;
        }
      }

      MPI_Isend(merged_result.data(), chunk_sizes[neighbor_rank] * sizeof(T), MPI_BYTE, neighbor_rank, 0,
                MPI_COMM_WORLD, &send_request);
      MPI_Wait(&send_request, MPI_STATUS_IGNORE);

      std::copy(merged_result.begin() + chunk_sizes[neighbor_rank],
                merged_result.begin() + chunk_sizes[neighbor_rank] + chunk_data.size(), chunk_data.begin());

    } else {
      MPI_Isend(chunk_data.data(), chunk_data.size() * sizeof(T), MPI_BYTE, neighbor_rank, 0, MPI_COMM_WORLD,
                &send_request);
      MPI_Wait(&send_request, MPI_STATUS_IGNORE);

      buffer.resize(chunk_data.size());
      MPI_Irecv(buffer.data(), chunk_data.size() * sizeof(T), MPI_BYTE, neighbor_rank, 0, MPI_COMM_WORLD,
                &recv_request);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      chunk_data = buffer;
    }
  }
  return true;
}

template <class T>
bool TestMPITaskParallel<T>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    master_data.resize(data_size);
    std::copy(reinterpret_cast<T*>(taskData->inputs[0]), reinterpret_cast<T*>(taskData->inputs[0]) + data_size,
              master_data.begin());
  }
  return true;
}

template <class T>
bool TestMPITaskParallel<T>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty() || taskData->inputs_count[0] <= 0 ||
        taskData->outputs_count[0] != data_size) {
      return false;
    }
  }
  return true;
}

template <class T>
bool TestMPITaskParallel<T>::run() {
  internal_order_test();
  boost::mpi::broadcast(world, data_size, 0);
  int chunk_size = data_size / world.size();
  std::vector<int> chunk_sizes(world.size(), chunk_size);
  std::vector<int> chunk_sizes_bytes(world.size(), chunk_size * sizeof(T));
  chunk_sizes[0] = (chunk_size + data_size % world.size());
  chunk_sizes_bytes[0] = (chunk_size + data_size % world.size()) * sizeof(T);

  if (world.rank() == 0)
    chunk_data.resize(chunk_sizes[0]);
  else
    chunk_data.resize(chunk_size);
  std::vector<int> displs(world.size(), 0);
  for (int i = 1; i < world.size(); i++) {
    displs[i] = displs[i - 1] + chunk_sizes_bytes[i - 1];
  }
  MPI_Scatterv(master_data.data(), chunk_sizes_bytes.data(), displs.data(), MPI_BYTE, chunk_data.data(),
               chunk_sizes_bytes[world.rank()], MPI_BYTE, 0, MPI_COMM_WORLD);
  bubble_sort();
  int neighbor_rank;
  for (int phase = 1; phase <= world.size(); phase++) {
    if (phase % 2 == 1) {
      neighbor_rank = world.rank() % 2 == 1 ? world.rank() - 1 : world.rank() + 1;
      chunk_merge_sort(neighbor_rank, chunk_sizes);
    } else {
      neighbor_rank = world.rank() % 2 == 1 ? world.rank() + 1 : world.rank() - 1;
      chunk_merge_sort(neighbor_rank, chunk_sizes);
    }
  }
  MPI_Gather(chunk_data.data(), chunk_data.size() * sizeof(T), MPI_BYTE, master_data.data(), chunk_size * sizeof(T),
             MPI_BYTE, 0, MPI_COMM_WORLD);
  return true;
}

template <class T>
bool TestMPITaskParallel<T>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(master_data.begin(), master_data.end(), reinterpret_cast<T*>(taskData->outputs[0]));
  }
  return true;
}

namespace chastov_v_bubble_sort {
template class TestMPITaskParallel<int>;
template class TestMPITaskParallel<double>;
}  // namespace chastov_v_bubble_sort