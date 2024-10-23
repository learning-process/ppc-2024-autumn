#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

namespace lopatin_i_count_words_mpi {

std::vector<char> generateLongString(int n) {
  std::vector<char> testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n; i++) {
    for (unsigned long int j = 0; j < testString.length(); j++) {
      testData.push_back(testString[j]);
    }
  }
  return testData;
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tempPtr[i];
  }
  wordCount = 0;
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  bool inWord = false;
  for (char c : input_) {
    if (c == ' ' || c == '\n') {
      inWord = false;
    } else if (!inWord) {
      wordCount++;
      inWord = true;
    }
  }
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int chunkSize = 0;
  if (world.rank() == 0) {
    input_ = std ::vector<char>(taskData->inputs_count[0]);
    auto* tmpPtr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned long int i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmpPtr[i];
    }
    chunkSize = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, chunkSize, 0);

  localInput_.resize(chunkSize);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * chunkSize, chunkSize);
    }
    localInput_ = std::vector<char>(input_.begin(), input_.begin() + chunkSize);
  } else {
    world.recv(0, 0, localInput_.data(), chunkSize);
  }
  wordCount = 0;
  localWordCount = 0;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  bool in_word = false;
  for (char c : localInput_) {
    if (c == ' ' || c == '\n') {
      in_word = false;
    } else if (!in_word) {
      localWordCount++;
      in_word = true;
    }
  }
  boost::mpi::reduce(world, localWordCount, wordCount, std::plus<>(), 0);
  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  }
  return true;
}

}  // namespace lopatin_i_count_words_mpi