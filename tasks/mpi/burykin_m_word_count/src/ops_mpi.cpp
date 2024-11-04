#include "mpi/burykin_m_word_count/include/ops_mpi.hpp"

using namespace std::chrono_literals;

namespace burykin_m_word_count {

bool TestTaskSequential::pre_processing() {
  internal_order_test();
  if (taskData->inputs[0] != nullptr && taskData->inputs_count[0] > 0) {
    input_ = std::string(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
  } else {
    input_ = "";
  }
  word_count_ = 0;
  return true;
}

bool TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
}

bool TestTaskSequential::run() {
  internal_order_test();
  word_count_ = count_words(input_);
  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = word_count_;
  return true;
}
bool TestTaskSequential::is_word_character(char c) { return std::isalpha(static_cast<unsigned char>(c)); }

int TestTaskSequential::count_words(const std::string &text) {
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

  // Init vectors
  length = taskData->inputs_count[0];

  if (world.rank() == 0) {
    input_ = std::vector<char>(length);
    char *tmp_ptr = reinterpret_cast<char *>(taskData->inputs[0]);
    for (int i = 0; i < length; i++) {
      input_[i] = tmp_ptr[i];
    }
    // Init values for output
    word_count_ = 0;
  }

  return true;
}

bool TestTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool TestTaskParallel::run() {
  internal_order_test();

  if (length == 0) {
    return true;
  }

  if (world.size() == 1) {
    for (int i = 0; i < length; i++) {
      if (!is_word_character(input_[i])) word_count_++;
    }
    word_count_++;
    return true;
  }

  int world_size = world.size();

  if (world.rank() > length) {
    return true;
  }
  if (world_size > length + 1) world_size = length + 1;

  int partSize = length / (world_size - 1);
  int endPartSize = length - partSize * (world_size - 2);

  if (world.rank() == 0) {
    for (int i = 0; i < world_size - 2; i++) {
      world.send(i + 1, 0, input_.data() + i * partSize, partSize);
    }
    world.send(world_size - 1, 0, input_.data() + (world_size - 2) * partSize, endPartSize);

    int counter = 0;
    for (int i = 0; i < world_size - 1; i++) {
      world.recv(i + 1, 1, &counter, 1);
      word_count_ += counter;
    }
    word_count_++;
  } else {
    int localPart = partSize;
    if (world_size - 1 == world.rank()) localPart = endPartSize;
    std::vector<char> chunk(localPart);
    int counter = 0;
    world.recv(0, 0, chunk.data(), localPart);

    for (char ch : chunk)
      if (!is_word_character(ch)) counter++;
    world.send(0, 1, &counter, 1);
  }
  return true;
}

bool TestTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = word_count_;
  }
  return true;
}

bool TestTaskParallel::is_word_character(char c) { return std::isalpha(static_cast<unsigned char>(c)); }

}  // namespace burykin_m_word_count
