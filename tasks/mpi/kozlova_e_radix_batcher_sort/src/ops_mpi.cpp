#include "mpi/kozlova_e_radix_batcher_sort/include/ops_mpi.hpp"

#include <boost/serialization/vector.hpp>
#include <cstring>
#include <vector>

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential::pre_processing() {
  internal_order_test();
  input_size = taskData->inputs_count[0];
  data.resize(input_size);
  auto* mas = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < input_size; i++) data[i] = mas[i];
  return true;
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] >= 0;
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential::run() {
  internal_order_test();
  radixSort(data);
  return true;
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < data.size(); i++) reinterpret_cast<double*>(taskData->outputs[0])[i] = data[i];
  return true;
}

void kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential::radixSort(std::vector<double>& a) {
  std::vector<uint64_t> bit_representation(a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t bits = *reinterpret_cast<uint64_t*>(&a[i]);
    if ((bits >> 63) != 0u) {
      bit_representation[i] = ~bits;
    } else {
      bit_representation[i] = bits | 0x8000000000000000;
    }
  }

  for (int bit = 0; bit < 64; ++bit) {
    std::vector<uint64_t> output(a.size());
    int count[2] = {0};

    for (size_t i = 0; i < bit_representation.size(); ++i) {
      count[(bit_representation[i] >> bit) & 1]++;
    }

    count[1] += count[0];

    for (int i = bit_representation.size() - 1; i >= 0; --i) {
      int val = (bit_representation[i] >> bit) & 1;
      output[--count[val]] = bit_representation[i];
    }

    bit_representation = output;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t bits = bit_representation[i];
    if ((bits & 0x8000000000000000) != 0u) {
      bits &= ~0x8000000000000000;
    } else {
      bits = ~bits;
    }
    memcpy(&a[i], &bits, sizeof(double));
  }
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
    input_.resize(input_size);
    auto* mas = reinterpret_cast<double*>(taskData->inputs[0]);
    for (int i = 0; i < input_size; i++) input_[i] = mas[i];
  }
  return true;
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI::validation() {
  internal_order_test();
  return world.rank() == 0 ? (taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] >= 0)
                           : true;
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI::run() {
  internal_order_test();
  boost::mpi::broadcast(world, input_size, 0);
  RadixSortWithOddEvenMerge(input_);
  return true;
}

bool kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < input_.size(); i++) output[i] = input_[i];
  }
  return true;
}

void kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI::radixSort(std::vector<double>& a) {
  std::vector<uint64_t> bit_representation(a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t bits = *reinterpret_cast<uint64_t*>(&a[i]);
    if ((bits >> 63) != 0u) {
      bit_representation[i] = ~bits;
    } else {
      bit_representation[i] = bits | 0x8000000000000000;
    }
  }

  for (int bit = 0; bit < 64; ++bit) {
    std::vector<uint64_t> output(a.size());
    int count[2] = {0};

    for (size_t i = 0; i < bit_representation.size(); ++i) {
      count[(bit_representation[i] >> bit) & 1]++;
    }

    count[1] += count[0];

    for (int i = bit_representation.size() - 1; i >= 0; --i) {
      int val = (bit_representation[i] >> bit) & 1;
      output[--count[val]] = bit_representation[i];
    }

    bit_representation = output;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t bits = bit_representation[i];
    if ((bits & 0x8000000000000000) != 0u) {
      bits &= ~0x8000000000000000;
    } else {
      bits = ~bits;
    }
    memcpy(&a[i], &bits, sizeof(double));
  }
}

void kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI::RadixSortWithOddEvenMerge(std::vector<double>& a) {
  a.resize(input_size);
  boost::mpi::broadcast(world, a.data(), a.size(), 0);

  int total_size = a.size();
  int num_processes = world.size();
  int local_size = total_size / num_processes;
  int remainder = total_size % num_processes;

  std::vector<int> send_counts(num_processes, local_size);
  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
  }

  std::vector<int> send_displacements(num_processes, 0);
  for (int i = 1; i < num_processes; ++i) {
    send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
  }

  std::vector<double> local_input(send_counts[world.rank()]);

  boost::mpi::scatterv(world, a.data(), send_counts, send_displacements, local_input.data(), send_counts[world.rank()],
                       0);

  radixSort(local_input);

  int step = 1;
  bool is_even_phase = true;

  while (step <= world.size()) {
    int partner = -1;
    radixSort(local_input);
    if (is_even_phase) {
      if (world.rank() % 2 == 0 && world.rank() + 1 < world.size()) {
        partner = world.rank() + 1;
      } else if (world.rank() % 2 != 0) {
        partner = world.rank() - 1;
      }
    } else {
      if (world.rank() % 2 != 0 && world.rank() + 1 < world.size()) {
        partner = world.rank() + 1;
      } else if (world.rank() % 2 == 0 && world.rank() > 0) {
        partner = world.rank() - 1;
      }
    }

    if (partner != -1) {
      std::vector<double> recv_data(local_input.size());
      world.sendrecv(partner, 0, local_input, partner, 0, recv_data);

      std::vector<double> merged(local_input.size() + recv_data.size());
      std::merge(local_input.begin(), local_input.end(), recv_data.begin(), recv_data.end(), merged.begin());

      radixSort(merged);

      int local_keep = send_counts[world.rank()];

      if (world.rank() < partner) {
        local_input.assign(merged.begin(), merged.begin() + local_keep);
      } else {
        local_input.assign(merged.end() - local_keep, merged.end());
      }
    }

    is_even_phase = !is_even_phase;
    step++;
  }

  boost::mpi::gatherv(world, local_input.data(), send_counts[world.rank()], a.data(), send_counts, send_displacements,
                      0);
}
