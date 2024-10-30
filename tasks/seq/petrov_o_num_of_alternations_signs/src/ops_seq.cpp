// Copyright 2024 Nesterov Alexander
#include "seq/petrov_o_num_of_alternations_signs/include/ops_seq.hpp"

using namespace std::chrono_literals;


bool petrov_o_num_of_alternations_signs_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  const auto count = taskData->inputs_count[0];

  // // Проверка на корректность типа данных во время компиляции.
  // static_assert(std::is_same_v<decltype(taskData->inputs[0]), ElementType*>, 
  //               "Type mismatch in taskData->inputs[0]. Expected ElementType*.");

  const int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  this->input_.resize(count);
  std::copy(input, input + count, std::begin(this->input_));

  this->res = 0;  //Обнуляем счетчик каждый новый запуск

  return true;
}



bool petrov_o_num_of_alternations_signs_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;  //Проверяем, что на выходе ожидается одно число
}

bool petrov_o_num_of_alternations_signs_seq::TestTaskSequential::run() {
  internal_order_test();

  if (input_.size() > 1) {
      for (size_t i = 1; i < input_.size(); i++) {
          if ((input_[i] < 0) ^ (input_[i - 1] < 0)) {
              this->res++;
          }
      }
  }

  return true;
}

bool petrov_o_num_of_alternations_signs_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;  //Передаем резульбтат
  return true;
}
