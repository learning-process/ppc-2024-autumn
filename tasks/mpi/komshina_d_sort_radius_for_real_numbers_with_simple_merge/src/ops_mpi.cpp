#include <mpi.h>
#include <algorithm>
#include <string>
#include <iterator>
#include <mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp>

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  input = std::vector<double>(input_ptr, input_ptr + taskData->inputs_count[0]);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] == 0) {
    return true;
  }

  if (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == taskData->inputs_count[0]) {
    return true;
  }

  return false;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  SortDouble(input);
  sort = input;
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(sort.begin(), sort.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (rank == 0) {
    auto* input_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    input = std::vector<double>(input_ptr, input_ptr + taskData->inputs_count[0]);
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::validation() {
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

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  ParallelSortDouble();
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (rank == 0) {
    auto* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(sort.begin(), sort.end(), output_ptr);
  }

  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::CountingSort(double* inp, double* out, int byteNum,
                                                                                 int size) {
  auto* mas = reinterpret_cast<unsigned char*>(inp);
  int counter[256] = {0};

  for (int i = 0; i < size; i++) {
    counter[mas[8 * i + byteNum]]++;
  }

  int tem = 0;
  for (int j = 0; j < 256; j++) {
    int b = counter[j];
    counter[j] = tem;
    tem += b;
  }

  for (int i = 0; i < size; i++) {
    out[counter[mas[8 * i + byteNum]]] = inp[i];
    counter[mas[8 * i + byteNum]]++;
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::SortDouble(std::vector<double>& data) {
  int size = data.size();
  if (size == 0) return;

  std::vector<double> out_positives;
  out_positives.reserve(size);
  std::vector<double> out_negatives;
  out_negatives.reserve(size);

  for (double num : data) {
    (num < 0 ? out_negatives : out_positives).push_back(num);
  }

  for (double& num : out_negatives) {
    num = -num;
  }

  std::vector<double> sorted_positives(out_positives.size());
  std::vector<double> sorted_negatives(out_negatives.size());

  double* inp_ptr;
  double* out_ptr;

  if (!out_positives.empty()) {
    inp_ptr = out_positives.data();
    out_ptr = sorted_positives.data();

    for (int i = 0; i < 8; i++) {
      CountingSort(inp_ptr, out_ptr, i, out_positives.size());
      std::swap(inp_ptr, out_ptr);
    }

    if (inp_ptr != out_positives.data()) {
      std::copy(inp_ptr, inp_ptr + out_positives.size(), out_positives.begin());
    }
  }

  if (!out_negatives.empty()) {
    inp_ptr = out_negatives.data();
    out_ptr = sorted_negatives.data();

    for (int i = 0; i < 8; i++) {
      CountingSort(inp_ptr, out_ptr, i, out_negatives.size());
      std::swap(inp_ptr, out_ptr);
    }

    if (inp_ptr != out_negatives.data()) {
      std::copy(inp_ptr, inp_ptr + out_negatives.size(), out_negatives.begin());
    }
  }

  for (double& num : out_negatives) {
    num = -num;
  }

  data.clear();
  data.insert(data.end(), out_negatives.rbegin(),
              out_negatives.rend());
  data.insert(data.end(), out_positives.begin(), out_positives.end());
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel::ParallelSortDouble() {
  int total_size;
  if (rank == 0) {
    total_size = input.size();
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_size = total_size / size;
  int remainder = total_size % size;
  int local_size = base_size + (rank < remainder ? 1 : 0);

  std::vector<double> local_data(local_size);
  std::vector<int> send_counts(size, base_size);
  std::vector<int> displacements(size, 0);

  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
  }

  for (int i = 1; i < size; ++i) {
    displacements[i] = displacements[i - 1] + send_counts[i - 1];
  }

  MPI_Scatterv(input.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(), local_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  SortDouble(local_data);

  std::vector<double> merged_data;
  if (rank == 0) {
    merged_data = local_data;
    std::vector<double> recv_buffer;
    MPI_Status status;

    for (int proc = 1; proc < size; ++proc) {
      recv_buffer.resize(send_counts[proc]);
      MPI_Recv(recv_buffer.data(), send_counts[proc], MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &status);

      std::vector<double> temp_merged;
      temp_merged.reserve(merged_data.size() + recv_buffer.size());
      std::merge(merged_data.begin(), merged_data.end(), recv_buffer.begin(), recv_buffer.end(),
                 std::back_inserter(temp_merged));
      merged_data.swap(temp_merged);
    }

    sort = merged_data;
  } else {
    MPI_Send(local_data.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}