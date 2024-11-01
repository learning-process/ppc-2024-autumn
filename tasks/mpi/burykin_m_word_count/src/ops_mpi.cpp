#include "mpi/burykin_m_word_count/include/ops_mpi.hpp"

namespace burykin_m_word_count {
//baba baab
bool TestTaskSequential::pre_processing() {
  internal_order_test();
  if (taskData->inputs[0] != nullptr && taskData->inputs_count[0] > 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  } else {
    input_ = "";
  }
  word_count_ = 0;
  return true;
}

bool TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 0 || taskData->inputs_count[0] > 0) && taskData->outputs_count[0] == 1;
}

bool TestTaskSequential::run() {
  internal_order_test();
  word_count_ = count_words(input_);
  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count_;
  return true;
}

bool TestTaskSequential::is_word_character(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }

int TestTaskSequential::count_words(const std::string& text) {
  int count = 0;
  bool in_word = false;

  for (char c : text) {
    if (is_word_character(c)) {
      if (!in_word) {
        count++;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }

  return count;
}

bool TestTaskParallel::pre_processing() {
  internal_order_test();
  unsigned int chunkSize = 0;
  if (world.rank() == 0) {
    input_ = std::vector<char>(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<char*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_.begin());

    chunkSize = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, chunkSize, 0);

  local_input_.resize(chunkSize);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * chunkSize, chunkSize);
    }
    std::copy(input_.begin(), input_.begin() + chunkSize, local_input_.begin());
  } else {
    world.recv(0, 0, local_input_.data(), chunkSize);
  }
  return true;
}

bool TestTaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestTaskParallel::run() {
  internal_order_test();
  bool in_word = false;
  local_word_count_ = 0;

  for (char c : local_input_) {
    if (is_word_character(c)) {
      if (!in_word) {
        local_word_count_++;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }

  boost::mpi::reduce(world, local_word_count_, word_count_, std::plus<>(), 0);
  return true;
}

bool TestTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count_;
  }
  return true;
}

bool TestTaskParallel::is_word_character(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }

int TestTaskParallel::count_words(const std::vector<char>& text) {
  int count = 0;
  bool in_word = false;

  for (char c : text) {
    if (is_word_character(c)) {
      if (!in_word) {
        count++;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }

  return count;
}

}  // namespace burykin_m_word_count