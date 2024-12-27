#include "mpi/kolokolova_d_radix_integer_merge_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

void kolokolova_d_radix_integer_merge_sort_mpi::counting_sort_radix(std::vector<int>& array, int degree) {
  int size_vector = int(array.size());
  std::vector<int> func_res(size_vector);
  std::vector<int> nums_of_digits(20, 0);

  for (int i = 0; i < size_vector; i++) {
    int index = (array[i] / degree) % 10;
    if (array[i] < 0) {
      index += 10;
    }
    nums_of_digits[index]++;
  }

  for (int i = 1; i < 20; i++) {
    nums_of_digits[i] += nums_of_digits[i - 1];
  }

  for (int i = size_vector - 1; i >= 0; i--) {
    int index = (array[i] / degree) % 10;
    if (array[i] < 0) {
      index += 10;
    }
    func_res[nums_of_digits[index] - 1] = array[i];
    nums_of_digits[index]--;
  }

  for (int i = 0; i < size_vector; i++) {
    array[i] = func_res[i];
  }
}

std::vector<int> kolokolova_d_radix_integer_merge_sort_mpi::merge_and_sort(const std::vector<int>& vec1,
                                                                           const std::vector<int>& vec2) {
  std::vector<int> sort_vector(vec1);
  sort_vector.insert(sort_vector.end(), vec2.begin(), vec2.end());
  sort_vector = radix_sort(sort_vector);
  for (int i = 0; i < int(sort_vector.size()); i++) {
  }
  return sort_vector;
}

std::vector<int> kolokolova_d_radix_integer_merge_sort_mpi::radix_sort(std::vector<int>& array) {
  int max_num = *max_element(array.begin(), array.end());
  int min_num = *min_element(array.begin(), array.end());

  // Process digits starting from the unit place and increasing the exponent
  for (int degree = 1; max_num / degree > 0 || min_num / degree < 0; degree *= 10) {
    // Call counting_sort_radix for the current exponent
    counting_sort_radix(array, degree);
  }

  std::vector<int> sorted_array;
  std::vector<int> negatives;
  std::vector<int> positives;

  // Separate numbers into negative and positive
  for (int num : array) {
    if (num < 0) {
      negatives.push_back(num);
    } else {
      positives.push_back(num);
    }
  }

  sort(negatives.begin(), negatives.end());
  sorted_array.insert(sorted_array.end(), negatives.begin(), negatives.end());
  sorted_array.insert(sorted_array.end(), positives.begin(), positives.end());

  return sorted_array;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_vector = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr_input = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_vector[i] = tmp_ptr_input[i];
  }
  res.resize(int(input_vector.size()));
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] != 0 && taskData->outputs_count[0] != 0);
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = radix_sort(input_vector);
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_ptr[i] = res[i];
  }
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_vector = std::vector<int>(taskData->inputs_count[0]);
    size_input_vector = taskData->inputs_count[0];
    auto* tmp_ptr_input = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_vector[i] = tmp_ptr_input[i];
    }
  }
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output and input
    return (taskData->inputs_count[0] != 0 && taskData->outputs_count[0] != 0);
  }
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int proc_rank = world.rank();
  int proc_size = world.size();

  broadcast(world, size_input_vector, 0);
  local_size = size_input_vector / proc_size;
  local_vector = std::vector<int>(local_size);
  merge_vec = std::vector<int>(local_size);
  remainder = size_input_vector % proc_size;
  if (remainder == 0)
    res.resize(size_input_vector);
  else
    res.resize(proc_size * local_size);

  // Send parts of vector for each proc
  if (proc_rank == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_vector.data() + proc * local_size, local_size);
    }
    if (remainder != 0) {
      std::copy(input_vector.begin() + proc_size * local_size, input_vector.end(), std::back_inserter(remaind_vector));
    }
  }
  if (proc_rank == 0) {
    local_vector = std::vector<int>(input_vector.begin(), input_vector.begin() + local_size);
  } else {
    world.recv(0, 0, local_vector.data(), local_size);
  }

  local_vector = radix_sort(local_vector);

  // Odd even merge sort
  for (int i = 0; i < proc_size; i++) {
    if (i % 2 == 0) {
      if (proc_rank % 2 == 0 && proc_rank + 1 < proc_size) {
        world.send(proc_rank + 1, proc_rank, local_vector.data(), local_size);
      } else if (proc_rank % 2 != 0) {
        world.recv(proc_rank - 1, proc_rank - 1, merge_vec.data(), local_size);
        std::vector<int> sort_vector = merge_and_sort(local_vector, merge_vec);
        std::copy(sort_vector.begin() + local_size, sort_vector.end(), local_vector.begin());
        world.send(proc_rank - 1, proc_rank, sort_vector.data(), local_size);
      }
      if (proc_rank % 2 == 0 && proc_rank + 1 < proc_size) {
        world.recv(proc_rank + 1, proc_rank + 1, local_vector.data(), local_size);
      }
    }
    if (i % 2 != 0) {
      if (proc_rank % 2 != 0 && proc_rank + 1 < proc_size) {
        world.send(proc_rank + 1, proc_rank, local_vector.data(), local_size);
      } else if (proc_rank % 2 == 0 && proc_rank > 0) {
        world.recv(proc_rank - 1, proc_rank - 1, merge_vec.data(), local_size);
        std::vector<int> sort_vector = merge_and_sort(local_vector, merge_vec);
        std::copy(sort_vector.begin() + local_size, sort_vector.end(), local_vector.begin());
        world.send(proc_rank - 1, proc_rank, sort_vector.data(), local_size);
      }
      if (proc_rank % 2 != 0 && proc_rank + 1 < proc_size) {
        world.recv(proc_rank + 1, proc_rank + 1, local_vector.data(), local_size);
      }
    }
  }

  gather(world, local_vector.data(), local_size, res, 0);

  if (proc_rank == 0 && remainder != 0) {
    res = merge_and_sort(res, remaind_vector);
  }

  return true;
}

bool kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t i = 0; i < res.size(); ++i) {
      output_ptr[i] = res[i];
    }
  }
  return true;
}
