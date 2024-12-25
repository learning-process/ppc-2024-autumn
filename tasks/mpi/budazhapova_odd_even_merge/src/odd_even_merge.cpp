#include "mpi/budazhapova_odd_even_merge/include/odd_even_merge.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace budazhapova_betcher_odd_even_merge_mpi {

void counting_sort(std::vector<int>& arr, int exp) {
  int n = arr.size();
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

  for (int i = 0; i < n; i++) {
    int index = (arr[i] / exp) % 10;
    count[index]++;
  }
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  for (int i = n - 1; i >= 0; i--) {
    int index = (arr[i] / exp) % 10;
    output[count[index] - 1] = arr[i];
    count[index]--;
  }
  for (int i = 0; i < n; i++) {
    arr[i] = output[i];
  }
}

void radix_sort(std::vector<int>& arr) {
  int max_num = *std::max_element(arr.begin(), arr.end());
  for (int exp = 1; max_num / exp > 0; exp *= 10) {
    counting_sort(arr, exp);
  }
}
}  // namespace budazhapova_betcher_odd_even_merge_mpi
bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::pre_processing() {
  internal_order_test();
  res = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
                         reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  n_el = taskData->inputs_count[0];
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::run() {
  internal_order_test();
  budazhapova_betcher_odd_even_merge_mpi::radix_sort(res);
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::post_processing() {
  internal_order_test();
  int* output = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); i++) {
    output[i] = res[i];
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::pre_processing() {
  internal_order_test();

  std::vector<int> recv_counts(world.size(), 0);
  std::vector<int> displacements(world.size(), 0);

  if (world.rank() == 0) {
    res = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
                           reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
    n_el = taskData->inputs_count[0];
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0;
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::run() {
  internal_order_test();

  std::vector<int> recv_counts(world.size(), 0);
  std::vector<int> displacements(world.size(), 0);

  boost::mpi::broadcast(world, res, 0);

  int n_of_send_elements;
  int n_of_proc_with_extra_elements;
  int start;
  int end;
  int world_size = world.size();
  int world_rank = world.rank();
  int res_size = static_cast<int>(res.size());

  for (int i = 0; i < world_size; i++) {
    n_of_send_elements = res_size / world_size;
    n_of_proc_with_extra_elements = res_size % world_size;

    start = i * n_of_send_elements + std::min(i, n_of_proc_with_extra_elements);
    end = start + n_of_send_elements + (i < n_of_proc_with_extra_elements ? 1 : 0);
    recv_counts[i] = end - start;
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + recv_counts[i - 1];
  }

  n_of_send_elements = res_size / world_size;
  n_of_proc_with_extra_elements = res_size % world_size;

  start = world_rank * n_of_send_elements + std::min(world_rank, n_of_proc_with_extra_elements);
  end = start + n_of_send_elements + (world_rank < n_of_proc_with_extra_elements ? 1 : 0);

  if (world.size() > res_size) {
    if (world.rank() < res_size) {
      local_res.resize(1);
      for (int i = start; i < end; i++) {
        for (int j = 0; j < n_of_send_elements; j++) {
          local_res[(i - start) * n_of_send_elements + j] = res[i * n_of_send_elements + j];
        }
      }
    } else {
      local_res.clear();
      return true;
    }
  } else {
    local_res.resize(end - start);
    for (int i = start; i < end; i++) {
      for (int j = 0; j < n_of_send_elements; j++) {
        local_res[(i - start) * n_of_send_elements + j] = res[i * n_of_send_elements + j];
      }
    }
  }

  budazhapova_betcher_odd_even_merge_mpi::radix_sort(local_res);

  for (int phase = 0; phase < world.size(); ++phase) {
    if (phase % 2 == 0) {
      if (world.rank() % 2 == 0 && world.rank() + 1 < world.size()) {
        boost::mpi::send(world, world.rank() + 1, 0, local_res);
      } else if (world.rank() % 2 == 1) {
        std::vector<int> received_data;
        boost::mpi::recv(world, world.rank() - 1, 0, received_data);
        local_res.insert(local_res.end(), received_data.begin(), received_data.end());
        budazhapova_betcher_odd_even_merge_mpi::radix_sort(local_res);
      }
    } else {
      if (world.rank() % 2 == 1 && world.rank() + 1 < world.size()) {
        boost::mpi::send(world, world.rank() + 1, 0, local_res);
      } else if (world.rank() % 2 == 0) {
        std::vector<int> received_data;
        boost::mpi::recv(world, world.rank() - 1, 0, received_data);
        local_res.insert(local_res.end(), received_data.begin(), received_data.end());
        budazhapova_betcher_odd_even_merge_mpi::radix_sort(local_res);
      }
    }
  }

  boost::mpi::gatherv(world, local_res.data(), local_res.size(), res.data(), recv_counts, displacements, 0);

  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::post_processing() {
  internal_order_test();
  int* output = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); i++) {
    output[i] = res[i];
  }
  return true;
}
