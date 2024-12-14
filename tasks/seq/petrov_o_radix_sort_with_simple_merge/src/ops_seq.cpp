#include "seq/petrov_o_radix_sort_with_simple_merge/include/ops_seq.hpp"

#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

namespace petrov_o_radix_sort_with_simple_merge_seq {

bool TestTaskSequential::validation() {
  internal_order_test();

  bool isValid = (!taskData->inputs_count.empty()) && 
                 (!taskData->inputs.empty()) && 
                 (!taskData->outputs.empty());

  return isValid;
}

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  input_.resize(size);

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(input_data, input_data + size, input_.begin());

  res.resize(size);
  return true;
}

bool petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential::run() {
    internal_order_test();

    // Инвертирование знакового бита для всех чисел
    for (auto& num : input_) {
        num ^= 0x80000000;
    }

    // Найти максимальное число для определения количества бит
    unsigned int max_num = static_cast<unsigned int>(input_[0]);
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

bool TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_[i] = res[i];
  }
  std::copy(res.begin(), res.end(), output_);

  return true;
}

}  // namespace petrov_o_radix_sort_with_simple_merge_seq