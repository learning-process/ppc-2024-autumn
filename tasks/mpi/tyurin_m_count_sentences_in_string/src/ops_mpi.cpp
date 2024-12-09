#include "mpi/tyurin_m_count_sentences_in_string/include/ops_mpi.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::pre_processing() {
  internal_order_test();
  input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  sentence_count_ = 0;
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::run() {
  internal_order_test();

  bool inside_sentence = false;
  for (char c : input_str_) {
    if (is_sentence_end(c)) {
      if (inside_sentence) {
        sentence_count_++;
        inside_sentence = false;
      }
    } else if (!is_whitespace(c)) {
      inside_sentence = true;
    }
  }
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = sentence_count_;
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::is_sentence_end(char c) {
  return c == '.' || c == '!' || c == '?';
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::is_whitespace(char c) {
  return c == ' ' || c == '\n' || c == '\t';
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  }

  local_sentence_count_ = 0;
  sentence_count_ = 0;

  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::run() {
  internal_order_test();

  size_t total_length{};
  if (world.rank() == 0) {
    total_length = input_str_.size();
  }
  boost::mpi::broadcast(world, total_length, 0);

  std::string local_segment;
  size_t segment_size = total_length / world.size();
  size_t remainder = 0;

  if (world.rank() == 0) {
    remainder = total_length % world.size();

    for (int rank = 1; rank < world.size(); rank++) {
      world.send(rank, 0, input_str_.data() + rank * segment_size + remainder, segment_size);
    }

    local_segment.assign(input_str_, 0, segment_size + remainder);
  } else {
    local_segment.resize(segment_size);
    world.recv(0, 0, local_segment.data(), segment_size);
  }

  bool in_sentence = false;

  for (char character : local_segment) {
    if (is_sentence_end(character)) {
      if (in_sentence || character == local_segment.front()) {
        local_sentence_count_++;
        in_sentence = false;
      }
    } else if (!is_whitespace(character)) {
      in_sentence = true;
    }
  }

  boost::mpi::reduce(world, local_sentence_count_, sentence_count_, std::plus<>(), 0);

  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count_;
  }

  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::is_sentence_end(char c) {
  return c == '.' || c == '!' || c == '?';
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::is_whitespace(char c) {
  return c == ' ' || c == '\n' || c == '\t';
}
