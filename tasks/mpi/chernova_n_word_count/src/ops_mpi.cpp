#include "mpi/chernova_n_word_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<char> chernova_n_word_count_mpi::clean_string(const std::vector<char>& input) {
  std::string result;
  std::string str(input.begin(), input.end());

  std::string::size_type pos = 0;
  while ((pos = str.find("  ", pos)) != std::string::npos) {
    str.erase(pos, 1);
  }

  pos = 0;
  while ((pos = str.find(" - ", pos)) != std::string::npos) {
    str.erase(pos, 2);
  }

  pos = 0;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  pos = str.size() - 1;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  result.assign(str.begin(), str.end());
  return std::vector<char>(result.begin(), result.end());
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  spaceCount = 0;
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  input_ = clean_string(input_);
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (input_.empty()) {
    spaceCount = -1;
  }
  for (std::size_t i = 0; i < input_.size(); i++) {
    char c = input_[i];
    if (c == ' ') {
      spaceCount++;
    }
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = spaceCount + 1;
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::vector<char>(taskData->inputs_count[0]);
    spaceCount = 0;
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (std::size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    input_ = clean_string(input_);
    taskData->inputs_count[0] = input_.size();
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned long totalSize = 0;
  if (world.rank() == 0) {
    totalSize = input_.size();
    partSize = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, partSize, 0);
  boost::mpi::broadcast(world, totalSize, 0);

  unsigned long startPos = world.rank() * partSize;
  unsigned long actualPartSize = (startPos + partSize <= totalSize) ? partSize : (totalSize - startPos);

  local_input_.resize(actualPartSize);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      unsigned long procStartPos = proc * partSize;
      unsigned long procPartSize = (procStartPos + partSize <= totalSize) ? partSize : (totalSize - procStartPos);
      if (procPartSize > 0) {
        world.send(proc, 0, input_.data() + procStartPos, procPartSize);
      }
    }
    local_input_.assign(input_.begin(), input_.begin() + actualPartSize);
  } else {
    if (actualPartSize > 0) {
      world.recv(0, 0, local_input_.data(), actualPartSize);
    }
  }
  localSpaceCount = 0;
  for (std::size_t i = 0; i < local_input_.size(); ++i) {
    if (local_input_[i] == ' ') {
      localSpaceCount++;
    }
  }

  boost::mpi::reduce(world, localSpaceCount, spaceCount, std::plus<>(), 0);

  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (spaceCount == 0) {
      spaceCount = -1;
    }
    reinterpret_cast<int*>(taskData->outputs[0])[0] = spaceCount + 1;
  }
  return true;
}
