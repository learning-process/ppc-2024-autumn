#include "mpi/petrov_o_radix_sort_with_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <limits>

namespace petrov_o_radix_sort_with_simple_merge_mpi {

bool TaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    bool isValid = (!taskData->inputs_count.empty()) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());
    return isValid;
  }

  return true;
}

bool TaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int size = taskData->inputs_count[0];
    input_.resize(size);
    int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(input_data, input_data + size, input_.begin());
    res.resize(size);
  }

  return true;
}

bool TaskParallel::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(input_.size());
  }
  boost::mpi::broadcast(world, n, 0);

  int base_count = n / size;
  int remainder = n % size;

  std::vector<int> send_counts(size, base_count);
  for (int i = 0; i < remainder; ++i) {
    send_counts[i] += 1;
  }

  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  std::vector<int> local_data(send_counts[rank]);

  if (rank == 0) {
    for (int proc = 1; proc < size; ++proc) {
      world.send(proc, 0, &input_[displs[proc]], send_counts[proc]);
    }
    std::copy(input_.begin(), input_.begin() + send_counts[0], local_data.begin());
  } else {
    world.recv(0, 0, local_data.data(), send_counts[rank]);
  }

  for (auto& num : local_data) {
    num ^= 0x80000000;
  }

  unsigned int local_max = 0;
  if (!local_data.empty()) {
    local_max = static_cast<unsigned int>(local_data[0]);
    for (auto& num : local_data) {
      auto val = static_cast<unsigned int>(num);
      if (val > local_max) {
        local_max = val;
      }
    }
  }

  unsigned int global_max = 0;
  boost::mpi::all_reduce(world, local_max, global_max, boost::mpi::maximum<unsigned int>());

  int num_bits = 0;
  const int MAX_BITS = static_cast<int>(sizeof(unsigned int) * 8);
  while (num_bits < MAX_BITS && (global_max >> num_bits) > 0) {
    num_bits++;
  }

  {
    std::vector<int> output(local_data.size());
    for (int bit = 0; bit < num_bits; ++bit) {
      int zero_count = 0;
      for (const auto& num : local_data) {
        if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
          zero_count++;
        }
      }

      int zero_index = 0;
      int one_index = zero_count;
      for (const auto& num : local_data) {
        if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
          output[zero_index++] = num;
        } else {
          output[one_index++] = num;
        }
      }
      local_data = output;
    }
  }

  for (auto& num : local_data) {
    num ^= 0x80000000;
  }

  std::vector<int> recv_buf;
  if (rank == 0) {
    recv_buf.resize(n);
    std::copy(local_data.begin(), local_data.end(), recv_buf.begin());
    for (int proc = 1; proc < size; ++proc) {
      world.recv(proc, 1, &recv_buf[displs[proc]], send_counts[proc]);
    }
  } else {
    world.send(0, 1, local_data.data(), send_counts[rank]);
  }

  if (rank == 0) {
    std::vector<int> final_result;
    final_result.insert(final_result.end(), recv_buf.begin(), recv_buf.begin() + send_counts[0]);

    auto merge_two = [](const std::vector<int>& a, const std::vector<int>& b) {
      std::vector<int> merged;
      merged.reserve(a.size() + b.size());
      size_t i = 0;
      size_t j = 0;
      while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
          merged.push_back(a[i++]);
        } else {
          merged.push_back(b[j++]);
        }
      }
      while (i < a.size()) {
        merged.push_back(a[i++]);
      }
      while (j < b.size()) {
        merged.push_back(b[j++]);
      }
      return merged;
    };

    for (int proc = 1; proc < size; ++proc) {
      std::vector<int> next_block(recv_buf.begin() + displs[proc], recv_buf.begin() + displs[proc] + send_counts[proc]);
      final_result = merge_two(final_result, next_block);
    }

    res = std::move(final_result);
  }

  return true;
}

bool TaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res.begin(), res.end(), output_);
  }
  return true;
}

/*---------------------------------------------SEQUENTIAL-----------------------------------------------------------*/

bool TaskSequential::validation() {
  internal_order_test();

  bool isValid = (!taskData->inputs_count.empty()) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());

  return isValid;
}

bool TaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  input_.resize(size);

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(input_data, input_data + size, input_.begin());

  res.resize(size);
  return true;
}

bool TaskSequential::run() {
  internal_order_test();

  for (auto& num : input_) {
    num ^= 0x80000000;
  }

  auto max_num = static_cast<unsigned int>(input_[0]);
  for (const auto& num : input_) {
    if (static_cast<unsigned int>(num) > max_num) {
      max_num = static_cast<unsigned int>(num);
    }
  }
  int num_bits = 0;
  const int MAX_BITS = sizeof(unsigned int) * 8;
  while (num_bits < MAX_BITS && (max_num >> num_bits) > 0) {
    num_bits++;
  }

  std::vector<int> output(input_.size());

  for (int bit = 0; bit < num_bits; ++bit) {
    int zero_count = 0;

    for (const auto& num : input_) {
      if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
        zero_count++;
      }
    }

    int zero_index = 0;
    int one_index = zero_count;

    for (const auto& num : input_) {
      if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
        output[zero_index++] = num;
      } else {
        output[one_index++] = num;
      }
    }

    input_ = output;
  }

  for (auto& num : input_) {
    num ^= 0x80000000;
  }

  res = input_;
  return true;
}

bool TaskSequential::post_processing() {
  internal_order_test();

  int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_[i] = res[i];
  }
  std::copy(res.begin(), res.end(), output_);

  return true;
}

}  // namespace petrov_o_radix_sort_with_simple_merge_mpi