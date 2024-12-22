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
  for (size_t i = 0; i < n_; ++i) {
    uint64_t u;
    std::memcpy(&u, &data_[i], sizeof(double));
    // Перевод для сохранения порядка
    if ((u & 0x8000000000000000ULL) != 0) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  radix_sort_uint64(keys);

  for (size_t i = 0; i < n_; ++i) {
    uint64_t u = keys[i];
    if ((u & 0x8000000000000000ULL) != 0) {
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

  int rank = world.rank();
  int size = world.size();
  int local_n = n / size;
  int remainder = n % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = local_n + (i < remainder ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  mpi::broadcast(world, counts, 0);
  mpi::broadcast(world, displs, 0);

  std::vector<double> local_data(counts[rank]);
  mpi::scatterv(world, (rank == 0 ? data.data() : (double*)nullptr), counts, displs, local_data.data(), counts[rank],
                0);

  // Локальная поразрядная сортировка
  radix_sort_doubles(local_data);

  // Организуем дерево слияний.
  // Общее число шагов = ceil(log2(size))
  int steps = 0;
  {
    int tmp = size;
    while (tmp > 1) {
      tmp = (tmp + 1) / 2;
      steps++;
    }
  }

  int group_size = 1;  // на шаге 0 объединяем пары, на шаге 1 объединяем пары по 2, и т.д.
  for (int step = 0; step < steps; ++step) {
    int partner_rank = rank + group_size;
    // Определяем, участвует ли текущий процесс в слиянии
    int group_step_size = group_size * 2;
    bool is_merger = (rank % group_step_size == 0);  // этот процесс будет принимать данные и сливать
    bool has_partner = (partner_rank < size);  // есть ли партнер для слияния

    if (is_merger && has_partner) {
      // Процесс принимает данные от partner_rank
      // Сначала получим размер массива партнёра
      int partner_size;
      world.recv(partner_rank, 0, partner_size);

      std::vector<double> partner_data(partner_size);
      world.recv(partner_rank, 1, partner_data.data(), partner_size);

      // Сливаем local_data и partner_data
      std::vector<double> merged;
      merged.reserve(local_data.size() + partner_data.size());
      std::merge(local_data.begin(), local_data.end(), partner_data.begin(), partner_data.end(),
                 std::back_inserter(merged));
      local_data.swap(merged);
    } else if (!is_merger && (rank % group_step_size == group_size)) {
      // Этот процесс отправляет данные тому, кто будет сливать
      int receiver = rank - group_size;
      int my_size = (int)local_data.size();
      world.send(receiver, 0, my_size);
      world.send(receiver, 1, local_data.data(), my_size);
      // После отправки данные этому процессу уже не нужны, он может очистить
      local_data.clear();
    }

    // После каждого шага половина процессов исключается из дальнейшего слияния
    // Удваиваем group_size
    group_size *= 2;

    // Все кроме тех, кто остался в текущем "дереве" слияния, могут закончить.
    // Но для простоты просто идём дальше. Процессы, отправившие данные, могут быть пустыми.
  }

  // После всех шагов у процесса с rank=0 будет весь отсортированный массив
  if (rank == 0) {
    data.swap(local_data);
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
    if ((u & 0x8000000000000000ULL) != 0) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  radix_sort_uint64(keys);

  for (size_t i = 0; i < n_; ++i) {
    uint64_t u = keys[i];
    if ((u & 0x8000000000000000ULL) != 0) {
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