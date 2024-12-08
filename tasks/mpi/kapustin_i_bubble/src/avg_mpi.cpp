#include "mpi/kapustin_i_bubble/include/avg_mpi.hpp"

std::vector<int> kapustin_i_bubble_sort_mpi::BubbleSortMPI::merge(int partner, std::vector<int>& local_data) {
  if (partner < 0 || partner >= world.size()) return local_data;

  std::vector<int> tmp;
  size_t send_size = local_data.size();
  size_t recv_size = 0;
  const int TAG_SIZE = 1;
  const int TAG_DATA = 2;
  if (world.rank() < partner) {
    world.send(partner, TAG_SIZE, &send_size, 1);
    world.recv(partner, TAG_SIZE, &recv_size, 1);
    tmp.resize(recv_size);
    world.send(partner, TAG_DATA, local_data.data(), send_size);
    world.recv(partner, TAG_DATA, tmp.data(), recv_size);
  } else {
    world.recv(partner, TAG_SIZE, &recv_size, 1);
    world.send(partner, TAG_SIZE, &send_size, 1);
    tmp.resize(recv_size);
    world.recv(partner, TAG_DATA, tmp.data(), recv_size);
    world.send(partner, TAG_DATA, local_data.data(), send_size);
  }

  std::vector<int> merged_result;
  size_t i = 0;
  size_t j = 0;
  while (i < local_data.size() && j < tmp.size()) {
    if (local_data[i] <= tmp[j]) {
      merged_result.push_back(local_data[i++]);
    } else {
      merged_result.push_back(tmp[j++]);
    }
  }
  while (i < local_data.size()) merged_result.push_back(local_data[i++]);
  while (j < tmp.size()) merged_result.push_back(tmp[j++]);

  size_t mid = merged_result.size() / 2;
  if (world.rank() < partner) {
    local_data.assign(merged_result.begin(), merged_result.begin() + mid);
  } else {
    local_data.assign(merged_result.begin() + mid, merged_result.end());
  }

  return local_data;
}

bool kapustin_i_bubble_sort_mpi::BubbleSortMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size_ = static_cast<int>(taskData->inputs_count[0]);
    auto* raw_data = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(raw_data, raw_data + size_);
  }
  return true;
}

bool kapustin_i_bubble_sort_mpi::BubbleSortMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) {
      return false;
    }
  }
  return true;
}

bool kapustin_i_bubble_sort_mpi::BubbleSortMPI::run() {
  internal_order_test();
  int total_elements = size_;
  boost::mpi::broadcast(world, total_elements, 0);

  int base_size = total_elements / world.size();
  int remainder = total_elements % world.size();
  int local_size = base_size + (world.rank() < remainder ? 1 : 0);
  std::vector<int> local_data(local_size);

  if (world.rank() == 0) {
    int offset = 0;
    for (int rank = 0; rank < world.size(); ++rank) {
      int chunk_size = base_size + (rank < remainder ? 1 : 0);
      if (rank == 0) {
        std::copy(input_.begin(), input_.begin() + chunk_size, local_data.begin());
      } else {
        world.send(rank, 0, &input_[offset], chunk_size);
      }
      offset += chunk_size;
    }
  } else {
    world.recv(0, 0, local_data.data(), local_size);
  }

  for (size_t i = 0; i < local_data.size(); ++i) {
    bool swapped = false;
    for (size_t j = 0; j < local_data.size() - i - 1; ++j) {
      if (local_data[j] > local_data[j + 1]) {
        std::swap(local_data[j], local_data[j + 1]);
        swapped = true;
      }
    }
    if (!swapped) break;
  }

  for (int step = 0; step < world.size(); ++step) {
    int partner;
    if (step % 2 == 0) {
      partner = (world.rank() % 2 == 0) ? world.rank() + 1 : world.rank() - 1;
    } else {
      partner = (world.rank() % 2 == 1) ? world.rank() + 1 : world.rank() - 1;
    }
    if (partner < 0 || partner >= world.size()) {
      world.barrier();
      continue;
    }
    local_data = merge(partner, local_data);
    world.barrier();
  }

  if (world.rank() == 0) {
    final_result = local_data;
    for (int rank = 1; rank < world.size(); ++rank) {
      int recv_size;
      world.recv(rank, 0, recv_size);
      std::vector<int> tmp(recv_size);
      world.recv(rank, 0, tmp.data(), recv_size);
      final_result.insert(final_result.end(), tmp.begin(), tmp.end());
    }
  } else {
    int send_size = local_data.size();
    world.send(0, 0, send_size);
    world.send(0, 0, local_data.data(), send_size);
  }

  return true;
}

bool kapustin_i_bubble_sort_mpi::BubbleSortMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    memcpy(taskData->outputs[0], final_result.data(), sizeof(int) * final_result.size());
  }

  return true;
}