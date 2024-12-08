// Copyright 2024 Nesterov Alexander
#include "seq/shishkarev_a_sum_of_vector_elements/include/ops_seq.hpp"

#include <numeric>

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::pre_processing() {
  internal_order_test();

  // Инициализация входных данных
  const auto input_size = taskData->inputs_count[0];
  input_data = std::vector<InOutType>(input_size);

  auto input_ptr = reinterpret_cast<InOutType*>(taskData->inputs[0]);
  std::copy(input_ptr, input_ptr + input_size, input_data.begin());

  // Инициализация результата
  result = InOutType{};
  return true;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::validation() {
  internal_order_test();

  // Проверка корректности входных и выходных данных
  const bool is_input_valid = taskData->inputs_count[0] > 0;
  const bool is_output_valid = taskData->outputs_count[0] == 1;
  return is_input_valid && is_output_valid;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::run() {
  internal_order_test();

  // Суммирование элементов входного вектора
  result = std::accumulate(input_data.begin(), input_data.end(), InOutType{});
  return true;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::post_processing() {
  internal_order_test();

  // Передача результата в выходной буфер
  auto output_ptr = reinterpret_cast<InOutType*>(taskData->outputs[0]);
  output_ptr[0] = result;
  return true;
}

// Эксплицитная инстанцизация шаблонов для используемых типов
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<double>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<uint8_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int64_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<float>;
