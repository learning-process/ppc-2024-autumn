#include "mpi/petrov_o_radix_sort_with_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <limits>

namespace petrov_o_radix_sort_with_simple_merge_mpi {

bool TaskParallel::validation() {
  internal_order_test();

  // Валидация данных только на процессе 0
  if (world.rank() == 0) {
    bool isValid = (!taskData->inputs_count.empty()) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());
    return isValid;
  }

    return true;
}

bool TaskParallel::pre_processing() {
  internal_order_test();

  // Данные доступны только процессу 0
  if (world.rank() == 0) {
    int size = taskData->inputs_count[0];
    input_.resize(size);
    int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(input_data, input_data + size, input_.begin());
    res.resize(size);
  }

  // На других процессах пока никаких данных
  return true;
}

bool TaskParallel::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  // Передаем размер массива от процесса 0 ко всем остальным
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(input_.size());
  }
  boost::mpi::broadcast(world, n, 0);

  // Распределяем данные по всем процессам
  // Определим размеры кусочков: деление на почти равные части
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
    // Отправляем части остальным процессам
    for (int proc = 1; proc < size; ++proc) {
      world.send(proc, 0, &input_[displs[proc]], send_counts[proc]);
    }
    // Свою часть копируем локально
    std::copy(input_.begin(), input_.begin() + send_counts[0], local_data.begin());
  } else {
    world.recv(0, 0, local_data.data(), send_counts[rank]);
  }

  // Теперь каждый процесс имеет свой local_data
  // Инвертируем знаковый бит
  // Это необхожимо для правильной обработки отрицательных чисел, ведущий бит которых всегда равен единице. После
  // инверитрования они естественным образом встанут на позиции перед положительными числами
  for (auto& num : local_data) {
    num ^= 0x80000000;
  }

  // Найти максимальный элемент локально
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

  // Определить глобальный максимум, чтобы все процессы знали максимальное число
  unsigned int global_max = 0;
  boost::mpi::all_reduce(world, local_max, global_max, boost::mpi::maximum<unsigned int>());

  // Определяем количество бит в максимальном числе
  int num_bits = 0;
  const int MAX_BITS = static_cast<int>(sizeof(unsigned int) * 8);
  while (num_bits < MAX_BITS && (global_max >> num_bits) > 0) {
    num_bits++;
  }

  // Поразрядная сортировка локальных данных
  // Реализация стандартная, как была в последовательном случае, но только для local_data
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

  // Восстановим знаковые биты
  for (auto& num : local_data) {
    num ^= 0x80000000;
  }

  // Теперь у каждого процесса есть локально отсортированная часть.
  // Соберём все отсортированные части на процесс 0 для слияния.
  // Сначала процесс 0 соберет все в один массив.

  std::vector<int> recv_buf;
  // Сначала соберём поэлементно размеры, но они у нас уже известны (send_counts)
  // Соберём данные обратно на процесс 0
  if (rank == 0) {
    recv_buf.resize(n);
    // Скопируем свою часть
    std::copy(local_data.begin(), local_data.end(), recv_buf.begin());
    // Получаем данные от остальных процессов
    for (int proc = 1; proc < size; ++proc) {
      world.recv(proc, 1, &recv_buf[displs[proc]], send_counts[proc]);
    }
  } else {
    // Отправляем свою часть на 0
    world.send(0, 1, local_data.data(), send_counts[rank]);
  }

  if (rank == 0) {
    // На процессе 0 есть все отсортированные блоки, нужно слить их в один массив.
    // Простой подход: последовательно сливать массивы.

    // У нас есть recv_buf, разбитый на скомпонованные сегменты:
    // Отрезки: [0:send_counts[0]), [displs[1]:displs[1]+send_counts[1]), ...
    // Сольём их всех:
    std::vector<int> final_result;
    // Начинаем с первого блока
    final_result.insert(final_result.end(), recv_buf.begin(), recv_buf.begin() + send_counts[0]);

    // Функция для слияния двух отсортированных массивов
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

    // Сливаем по очереди оставшиеся блоки
    for (int proc = 1; proc < size; ++proc) {
      std::vector<int> next_block(recv_buf.begin() + displs[proc], recv_buf.begin() + displs[proc] + send_counts[proc]);
      final_result = merge_two(final_result, next_block);
    }

    // final_result теперь содержит полностью отсортированный массив
    res = std::move(final_result);
  }

  return true;
}

bool TaskParallel::post_processing() {
  internal_order_test();
  // Только процесс 0 записывает результат
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

  // Инвертирование знакового бита для всех чисел
  for (auto& num : input_) {
    num ^= 0x80000000;
  }

  // Найти максимальное число для определения количества бит
  auto max_num = static_cast<unsigned int>(input_[0]);
  for (const auto& num : input_) {
    if (static_cast<unsigned int>(num) > max_num) {
      max_num = static_cast<unsigned int>(num);
    }
  }
  // Определить количество бит в максимальном числе
  int num_bits = 0;
  const int MAX_BITS = sizeof(unsigned int) * 8;
  while (num_bits < MAX_BITS && (max_num >> num_bits) > 0) {
    num_bits++;
  }

  // Инициализация вспомогательного массива
  std::vector<int> output(input_.size());

  // Поразрядная сортировка
  for (int bit = 0; bit < num_bits; ++bit) {
    int zero_count = 0;

    // Подсчёт нулевых битов на текущем разряде
    for (const auto& num : input_) {
      if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
        zero_count++;
      }
    }

    // Индексы для размещения чисел
    int zero_index = 0;
    int one_index = zero_count;

    // Размещение чисел в output на основе текущего бита
    for (const auto& num : input_) {
      if (((static_cast<unsigned int>(num) >> bit) & 1) == 0) {
        output[zero_index++] = num;
      } else {
        output[one_index++] = num;
      }
    }

    // Копирование отсортированных данных обратно в input_ для следующей итерации
    input_ = output;
  }

  // Восстановление исходных значений путём инвертирования знакового бита
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