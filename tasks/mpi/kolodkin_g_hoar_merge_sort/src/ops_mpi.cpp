#include "mpi/kolodkin_g_hoar_merge_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <vector>

int partition(std::vector<int>& arr, int low, int high) {
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j < high; j++) {
    if (arr[j] <= pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return (i + 1);
}

void quickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  output_ = std::vector<int>(taskData->inputs_count[0]);
  output_ = input_;
  return true;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  quickSort(output_, 0, output_.size() - 1);
  return true;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  return true;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    output_ = std::vector<int>(taskData->inputs_count[0]);
    output_ = input_;
  }
  return true;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 1;
  }
  return true;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int num_processes = world.size();
  std::vector<int> send_counts(num_processes, 0);
  std::vector<int> displacements(num_processes, 0);

  if (world.rank() == 0) {
    send_counts.resize(num_processes);
    displacements.resize(num_processes);

    for (size_t i = 0; i < num_processes; i++) {
      send_counts[i] = input_.size() / num_processes;
      if (i == num_processes - 1) {
        send_counts[i] += input_.size() % num_processes;
      }
    }

    for (size_t i = 1; i < num_processes; i++) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }
  }
  for (size_t i = 0; i < num_processes; i++) {
    boost::mpi::broadcast(world, send_counts[i], 0);
    boost::mpi::broadcast(world, displacements[i], 0);
  }
  world.barrier();
  std::vector<int> local_input_(send_counts[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), send_counts, displacements, local_input_.data(), local_input_.size(), 0);

  quickSort(local_input_, 0, local_input_.size() - 1);

  std::vector<int> sorted_data;
  if (world.rank() == 0) {
    sorted_data.resize(input_.size());
  }

  boost::mpi::gatherv(world, local_input_.data(), local_input_.size(), sorted_data.data(), send_counts, displacements,
                      0);
  if (world.rank() == 0) {
    std::vector<int> final_result;
    std::vector<int> current_indices(num_processes, 0);
    std::vector<bool> done(num_processes, false);
    while (final_result.size() < input_.size()) {
      int min_index = -1;
      int min_value = std::numeric_limits<int>::max();

      for (size_t i = 0; i < num_processes; i++) {
        if (!done[i] && current_indices[i] < send_counts[i]) {
          int value = sorted_data[displacements[i] + current_indices[i]];
          if (value < min_value) {
            min_value = value;
            min_index = i;
          }
        }
      }
      if (min_index == -1) {
        break;
      }

      final_result.push_back(min_value);
      current_indices[min_index]++;
    }
    output_ = final_result;
  }
  return true;
}

bool kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  }
  return true;
}
