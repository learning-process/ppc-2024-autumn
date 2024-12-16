#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

namespace lopatin_i_count_words_mpi {

std::vector<char> generateLongString(int n) {
  std::vector<char> testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n - 1; i++) {
    for (unsigned long int j = 0; j < testString.length(); j++) {
      testData.push_back(testString[j]);
    }
  }
  std::string lastSentence = "This is a long sentence for performance testing of the word count algorithm using MPI.";
  for (unsigned long int j = 0; j < lastSentence.length(); j++) {
    testData.push_back(lastSentence[j]);
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
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  for (char c : input_) {
    if (c == ' ') {
      spaceCount++;
    }
  }
  wordCount = spaceCount + 1;
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::vector<char>(taskData->inputs_count[0]);
    auto* tmpPtr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned long int i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmpPtr[i];
    }
  }
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  unsigned long totalSize = 0;
  if (world.rank() == 0) {
    totalSize = input_.size();
    chunkSize = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, chunkSize, 0);
  boost::mpi::broadcast(world, totalSize, 0);

  unsigned long startPos = world.rank() * chunkSize;
  unsigned long actualChunkSize = (startPos + chunkSize <= totalSize) ? chunkSize : (totalSize - startPos);

  localInput_.resize(actualChunkSize);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      unsigned long procStartPos = proc * chunkSize;
      unsigned long procChunkSize = (procStartPos + chunkSize <= totalSize) ? chunkSize : (totalSize - procStartPos);
      if (procChunkSize > 0) {
        world.send(proc, 0, input_.data() + procStartPos, procChunkSize);
      }
    }
    localInput_.assign(input_.begin(), input_.begin() + actualChunkSize);
  } else {
    if (actualChunkSize > 0) {
      world.recv(0, 0, localInput_.data(), actualChunkSize);
    }
  }

  localSpaceCount = 0;
  for (size_t i = 0; i < localInput_.size(); ++i) {
    if (localInput_[i] == ' ') {
      localSpaceCount++;
    }
  }

  boost::mpi::reduce(world, localSpaceCount, spaceCount, std::plus<>(), 0);

  if (world.rank() == 0) {
    wordCount = spaceCount + 1;
  }
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