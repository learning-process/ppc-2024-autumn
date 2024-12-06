#include "mpi/kharin_m_radix_double_sort/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>
#include <queue>

namespace mpi = boost::mpi;
using namespace kharin_m_radix_double_sort;

bool RadixSortSequential::pre_processing() {
  internal_order_test();

  // Считываем данные
  data.resize(n);
  auto* arr = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(arr, arr + n, data.begin());

  return true;
}

bool RadixSortSequential::validation() {
  internal_order_test();

  bool is_valid = true;
  // Проверяем, что n и количество данных соответствуют
  n = *(reinterpret_cast<int*>(taskData->inputs[0]));
  if (taskData->inputs_count[0] != 1 || taskData->inputs_count[1] != static_cast<size_t>(n) ||
      taskData->outputs_count[0] != static_cast<size_t>(n)) {
    is_valid = false;
  }

  return is_valid;
}

bool RadixSortSequential::run() {
  internal_order_test();

  // Поразрядная сортировка
  radix_sort_doubles(data);
  return true;
}

bool RadixSortSequential::post_processing() {
  internal_order_test();

  // Записываем результат
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(data.begin(), data.end(), out);
  return true;
}

void RadixSortSequential::radix_sort_doubles(std::vector<double>& data_) {
  size_t n_ = data_.size();
  std::vector<uint64_t> keys(n_);
  for (size_t i = 0; i < n; ++i) {
    uint64_t u;
    std::memcpy(&u, &data_[i], sizeof(double));
    // Перевод для сохранения порядка
    if (u & 0x8000000000000000ULL) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  radix_sort_uint64(keys);

  for (size_t i = 0; i < n_; ++i) {
    uint64_t u = keys[i];
    if (u & 0x8000000000000000ULL) {
      u &= ~0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    std::memcpy(&data_[i], &u, sizeof(double));
  }
}

void RadixSortSequential::radix_sort_uint64(std::vector<uint64_t>& keys) {
  const int BITS = 64;
  const int RADIX = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < BITS; shift += 8) {
    size_t count[RADIX + 1] = {0};
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      ++count[byte + 1];
    }
    for (int i = 0; i < RADIX; ++i) {
      count[i + 1] += count[i];
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      temp[count[byte]++] = keys[i];
    }
    keys.swap(temp);
  }
}

bool RadixSortParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Считываем n
    data.resize(n);
    auto* arr = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(arr, arr + n, data.begin());
  }

  return true;
}

bool RadixSortParallel::validation() {
  internal_order_test();

  bool is_valid = true;
  if (world.rank() == 0) {
    n = *(reinterpret_cast<int*>(taskData->inputs[0]));
    // Проверяем размеры
    if (taskData->inputs_count[0] != 1 || taskData->inputs_count[1] != static_cast<size_t>(n) ||
        taskData->outputs_count[0] != static_cast<size_t>(n)) {
      is_valid = false;
    }
  }
  mpi::broadcast(world, is_valid, 0);
  mpi::broadcast(world, n, 0);
  return is_valid;
}

bool RadixSortParallel::run() {
  internal_order_test();

  // Распространяем данные
  if (world.rank() != 0) {
    data.resize(n);
  }
  mpi::broadcast(world, data.data(), n, 0);

  // Делим данные между процессами
  int size = world.size();
  int rank = world.rank();

  int local_n = n / size;
  int remainder = n % size;

  std::vector<int> counts(size), displs(size);
  for (int i = 0; i < size; ++i) {
    counts[i] = local_n + (i < remainder ? 1 : 0);
  }
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<double> local_data(counts[rank]);
  // Scatter данных
  mpi::scatterv(world, data.data(), counts, displs, local_data.data(), counts[rank], 0);

  // Локальная поразрядная сортировка
  radix_sort_doubles(local_data);

  // Сбор отсортированных данных
  std::vector<double> gathered;
  if (rank == 0) {
    gathered.resize(n);
  }
  mpi::gatherv(world, local_data.data(), counts[rank], gathered.data(), counts, displs, 0);

  // Слияние отсортированных подмассивов на процессе 0
  if (rank == 0) {
    // Разбиваем gathered по подмассивам
    std::vector<std::vector<double>> subarrays(size);
    for (int i = 0; i < size; ++i) {
      subarrays[i].resize(counts[i]);
      std::copy(gathered.begin() + displs[i], gathered.begin() + displs[i] + counts[i], subarrays[i].begin());
    }

    data = merge_sorted_subarrays(subarrays);
  }

  return true;
}

bool RadixSortParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(data.begin(), data.end(), out);
  }

  return true;
}

void RadixSortParallel::radix_sort_doubles(std::vector<double>& data_) {
  size_t n_ = data_.size();
  std::vector<uint64_t> keys(n_);
  for (size_t i = 0; i < n_; ++i) {
    uint64_t u;
    std::memcpy(&u, &data_[i], sizeof(double));
    if (u & 0x8000000000000000ULL) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  radix_sort_uint64(keys);

  for (size_t i = 0; i < n_; ++i) {
    uint64_t u = keys[i];
    if (u & 0x8000000000000000ULL) {
      u &= ~0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    std::memcpy(&data_[i], &u, sizeof(double));
  }
}

void RadixSortParallel::radix_sort_uint64(std::vector<uint64_t>& keys) {
  const int BITS = 64;
  const int RADIX = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < BITS; shift += 8) {
    size_t count[RADIX + 1] = {0};
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      ++count[byte + 1];
    }
    for (int i = 0; i < RADIX; ++i) {
      count[i + 1] += count[i];
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      temp[count[byte]++] = keys[i];
    }
    keys.swap(temp);
  }
}

std::vector<double> RadixSortParallel::merge_sorted_subarrays(
    const std::vector<std::vector<double>>& sorted_subarrays) {
  typedef std::pair<double, std::pair<int, int>> HeapNode; 
  std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<>> min_heap;

  // Инициализация кучи
  for (int i = 0; i < (int)sorted_subarrays.size(); ++i) {
    if (!sorted_subarrays[i].empty()) {
      min_heap.push({sorted_subarrays[i][0], {i, 0}});
    }
  }

  std::vector<double> result;
  result.reserve(n);
  while (!min_heap.empty()) {
    auto node = min_heap.top();
    min_heap.pop();
    double value = node.first;
    int array_idx = node.second.first;
    int elem_idx = node.second.second;
    result.push_back(value);
    if (elem_idx + 1 < (int)sorted_subarrays[array_idx].size()) {
      min_heap.push({sorted_subarrays[array_idx][elem_idx + 1], {array_idx, elem_idx + 1}});
    }
  }

  return result;
}