#include "mpi/kharin_m_number_of_sentences_mpi/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>

namespace kharin_m_number_of_sentences_mpi {

bool CountSentencesSequential::pre_processing() {
  internal_order_test();
  text = std::string(reinterpret_cast<const char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  sentence_count = 0;
  return true;
}

bool CountSentencesSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool CountSentencesSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < text.size(); i++) {
    char c = text[i];
    if (c == '.' || c == '?' || c == '!') {
      sentence_count++;
    }
  }
  return true;
}

bool CountSentencesSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  return true;
}

bool CountSentencesParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    text = std::string(reinterpret_cast<const char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  }
  sentence_count = 0;
  return true;
}

bool CountSentencesParallel::validation() {
  internal_order_test();
  return world.rank() == 0 ? taskData->outputs_count[0] == 1 : true;
}

bool CountSentencesParallel::run() {
  internal_order_test();
  int base_part_size = 0;
  int remainder = 0;
  int text_length = 0;
  if (world.rank() == 0) {
    text_length = text.size();
    base_part_size = text_length / world.size();
    remainder = text_length % world.size();
  }

  boost::mpi::broadcast(world, base_part_size, 0);
  boost::mpi::broadcast(world, remainder, 0);

  // Вычисляем начальную и конечную позиции для каждого процесса
  int start = world.rank() * base_part_size + std::min(world.rank(), remainder);
  int end = start + base_part_size + (world.rank() < remainder ? 1 : 0);

  // Каждый процесс создает свою local_text
  int delta = end - start;
  local_text = std::string(delta, ' ');
  copy(reinterpret_cast<const char*>(taskData->inputs[0]) + start,
       reinterpret_cast<const char*>(taskData->inputs[0]) + end, local_text.begin());

  // Подсчет предложений в локальной части
  int local_count = 0;
  for (size_t i = 0; i < local_text.size(); i++) {
    char c = local_text[i];
    if (c == '.' || c == '?' || c == '!') {
      local_count++;
    }
  }

  // Суммирование результатов
  boost::mpi::reduce(world, local_count, sentence_count, std::plus<>(), 0);
  return true;
}

bool CountSentencesParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  }
  return true;
}

}  // namespace kharin_m_number_of_sentences_mpi