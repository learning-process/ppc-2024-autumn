#include <mpi/sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging/include/ops_mpi.hpp>

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  input_data_.assign(input_ptr, input_ptr + taskData->inputs_count[0]);
  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return true;
  }
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  sequentialSort();

  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  if (sorted_data_.empty()) {
    return false;
  }
  std::copy(sorted_data_.begin(), sorted_data_.end(), output_ptr);
  return true;
}

void sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskSequential::sequentialSort() {
  sorted_data_ = input_data_;
  radixSortWithSignHandling(sorted_data_);
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (rank == 0) {
    auto* input_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    input_data_.assign(input_ptr, input_ptr + taskData->inputs_count[0]);

    std::sort(input_data_.rbegin(), input_data_.rend());
  }
  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (rank == 0) {
    if (taskData->inputs_count.size() != taskData->outputs_count.size()) {
      return false;
    }

    for (size_t i = 0; i < taskData->inputs_count.size(); ++i) {
      if (taskData->inputs_count[i] != taskData->outputs_count[i]) {
        return false;
      }
    }
    return true;
  }
  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  parallelSort();
  return true;
}

bool sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (rank == 0) {
    auto* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(sorted_data_.begin(), sorted_data_.end(), output_ptr);
  }

  return true;
}

void sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::radixSortWithSignHandling(
    std::vector<double>& data) {
  const int num_bits = sizeof(double) * 8;
  const int radix = 2;

  std::vector<double> positives;
  std::vector<double> negatives;

  for (double num : data) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }

  radixSort(positives, num_bits, radix);
  radixSort(negatives, num_bits, radix);

  for (double& num : negatives) {
    num = -num;
  }

  data.clear();
  data.insert(data.end(), negatives.rbegin(), negatives.rend());
  data.insert(data.end(), positives.begin(), positives.end());
}

void sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::radixSort(std::vector<double>& data,
                                                                                     int num_bits, int radix) {
  std::vector<std::vector<double>> buckets(radix);
  std::vector<double> output(data.size());

  for (int exp = 0; exp < num_bits; ++exp) {
    for (auto& num : data) {
      uint64_t bits = *reinterpret_cast<uint64_t*>(&num);
      int digit = (bits >> exp) & 1;
      buckets[digit].push_back(num);
    }

    int index = 0;
    for (int i = 0; i < radix; ++i) {
      for (auto& num : buckets[i]) {
        output[index++] = num;
      }
      buckets[i].clear();
    }

    data = output;
  }
}

void sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel::parallelSort() {
  int total_size;
  if (rank == 0) total_size = input_data_.size();
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  int base_size = total_size / size;
  int remainder = total_size % size;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displacements[i] = (i > 0) ? (displacements[i - 1] + send_counts[i - 1]) : 0;
  }

  std::vector<double> local_data(send_counts[rank]);
  MPI_Scatterv(input_data_.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(),
               send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  radixSortWithSignHandling(local_data);

  if (rank == 0) {
    std::vector<double> recv_data(base_size + 1);
    std::vector<double> merged_data;
    sorted_data_ = local_data;

    MPI_Status status;
    merged_data.reserve(total_size);
    for (int proc = 1; proc < size; proc++) {
      MPI_Recv(recv_data.data(), send_counts[proc], MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &status);
      std::merge(sorted_data_.begin(), sorted_data_.end(), recv_data.begin(), recv_data.begin() + send_counts[proc],
                 std::back_inserter(merged_data));
      sorted_data_ = merged_data;
      merged_data.clear();
    }
  } else {
    MPI_Send(local_data.data(), local_data.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}
