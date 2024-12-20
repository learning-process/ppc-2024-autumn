// Copyright 2023 Nesterov Alexander
// здесь писать саму задачу
#include "mpi/zolotareva_a_count_of_words/include/ops_mpi.hpp"

#include <string>

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_.assign(reinterpret_cast<char *>(taskData->inputs[0]),
                taskData->inputs_count[0]);
  res = 0;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  bool in_word = false;
  for (char c : input_) {
    if (c == ' ' && in_word) {
      ++res;
      in_word = false;
    } else if (c != ' ') {
      in_word = true;
    }
  }
  if (in_word)
    ++res;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);
  }
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int str_size = static_cast<int>(taskData->inputs_count[0]);
    input_.assign(reinterpret_cast<char *>(taskData->inputs[0]), str_size);
  }
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int str_size = 0;
  if (world.rank() == 0) {
    str_size = static_cast<int>(input_.size());
  }

  // Рассылаем длину строки всем процессам
  boost::mpi::broadcast(world, str_size, 0);

  int world_size = world.size();
  int rank = world.rank();

  int delta = str_size / world_size;
  int remainder = str_size % world_size;

  int start = 0;
  int length = 0;

  if (rank == 0) {
    start = 0;
    length = delta + remainder;
  } else {
    start = remainder + rank * delta;
    length = delta;
  }

  // Рассылка данных от процесса 0 к остальным процессам
  // Если у нас больше одного процесса
  if (world_size > 1) {
    if (rank == 0) {
      // Процесс 0 уже имеет свой кусок
      local_input_ = input_.substr(start, length);

      // Отправляем остальным
      for (int proc = 1; proc < world_size; proc++) {
        int proc_start = remainder + proc * delta;
        int proc_length = delta;
        if (proc_length > 0) {
          world.send(proc, 0, input_.data() + proc_start, proc_length);
        }
      }
    } else {
      if (length > 0) {
        local_input_ = std::string(length, '\0');
        world.recv(0, 0, &local_input_[0], length);
      } else {
        local_input_.clear();
      }
    }
  } else {
    local_input_ = input_;
  }

  // Подсчет слов в локальном фрагменте
  int local_res = 0;
  bool in_word = false;
  for (char c : local_input_) {
    if (c == ' ' && in_word) {
      ++local_res;
      in_word = false;
    } else if (c != ' ') {
      in_word = true;
    }
  }
  if (in_word)
    ++local_res;

  // Сбор данных от всех процессов
  char first_char = (local_input_.empty() ? ' ' : local_input_.front());
  char last_char = (local_input_.empty() ? ' ' : local_input_.back());

  std::vector<int> all_counts;
  std::vector<char> all_first_chars;
  std::vector<char> all_last_chars;

  boost::mpi::all_gather(world, local_res, all_counts);
  boost::mpi::all_gather(world, first_char, all_first_chars);
  boost::mpi::all_gather(world, last_char, all_last_chars);

  if (rank == 0) {
    int total = 0;
    for (int c : all_counts) {
      total += c;
    }

    // Коррекция разрывов слов на границах
    for (int i = 0; i < (world_size - 1); i++) {
      if (all_last_chars[i] != ' ' && all_first_chars[i + 1] != ' ') {
        total -= 1;
      }
    }

    res = total;
  }

  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  }
  return true;
}
