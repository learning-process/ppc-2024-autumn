#include "mpi/chernova_n_word_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
/*
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
*/
bool chernova_n_word_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  //std::cout << "-----------------------------------------!!!------------------------" << std::endl;
  input_ = std::vector<char>(taskData->inputs_count[0]);
  //spaceCount = 0;
  letter = 0;
  wordCount = 0;
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  //input_ = clean_string(input_);
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  /*
  if (input_.empty()) {
    spaceCount = -1;
  }
  for (std::size_t i = 0; i < input_.size(); i++) {
    char c = input_[i];
    if (c == ' ') {
      spaceCount++;
    }
  }
  */
  for (size_t i = 0; i < input_.size(); i++) {
    char c = input_[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      letter++;
    } else {
      if (c == ' ') {
        if (letter > 0) {
          wordCount++;
          letter = 0;
        }
      }
    }
  }
  if (letter > 0) {
    wordCount++;
    letter = 0;
  }
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::vector<char>(taskData->inputs_count[0]);
    letter = 0;
    wordCount = 0;
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (std::size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    //input_ = clean_string(input_);
    //taskData->inputs_count[0] = input_.size();
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
  std::cout << "-----------------------------------------!!!------------------------";
  unsigned long totalSize = 0;
  int fakeWord=0;
  if (world.rank() == 0) {
    totalSize = input_.size();
    partSize = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, partSize, 0);
  boost::mpi::broadcast(world, totalSize, 0);

 // unsigned long startPos = world.rank() * partSize;  // abcdef
  unsigned long actualPartSize = partSize;
  //(startPos + partSize > totalSize) ? partSize : (totalSize - startPos);

  //local_input_.resize(actualPartSize);

  if (world.rank() == 0) {
  unsigned long mod = totalSize % world.size();
      actualPartSize = partSize+mod;

    for (int proc = 1; proc < world.size(); proc++) {
      unsigned long procStartPos = proc * partSize + mod;
      //unsigned long procPartSize = partSize;
      //(procStartPos + partSize <= totalSize) ? partSize : (totalSize - procStartPos);

      if (input_[procStartPos] != ' ' && ((input_[procStartPos - 1] >= 'a' && input_[procStartPos - 1] <= 'z') ||
          (input_[procStartPos - 1] >= 'A' && input_[procStartPos - 1] <= 'Z'))) {
          fakeWord++;}
      
      //if (procPartSize > 0) {
      //world.send(proc, 0, input_.data() + procStartPos, partSize);
      world.send(proc, 0, input_.data() + procStartPos, partSize+1);
      std::cout << "procpartsize " << partSize << std::endl;
      //}
      std::cout << input_.data() + procStartPos << " " << proc << " " << partSize << " " << std::endl;
    }
    local_input_ = std::vector<char>(actualPartSize);
    //for (size_t i=0;i<local_input_.size)
    local_input_.assign(input_.begin(), input_.begin() + actualPartSize);
    std::cout << local_input_.data()<< " 0 " << " " << actualPartSize << " " << std::endl;
  } else {
    //if (actualPartSize > 0) {
    local_input_ = std::vector<char>(actualPartSize);
    std::cout << actualPartSize << std::endl;
    //world.recv(0, 0, local_input_.data(), actualPartSize);
    std::cout << "I AM HERE!!!!!!!!!!!!!!!!!!!!!!!" << local_input_.size() << " " << world.rank() << std::endl << std::endl;
    local_input_.resize(partSize);
      world.recv(0, 0, local_input_.data(), actualPartSize);
      std::cout << local_input_.data()<< " " << world.rank() << " Proceess said " << std::endl;
    //}
      //aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    //for (size_t i = 0; i < local_input_.size())
  }
  local_input_.resize(actualPartSize);
  localWordCount = 0;
  letter = 0;
  for (size_t i = 0; i < actualPartSize; i++) {
    char c = local_input_[i];
    //std::cout << c;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      letter++;
      //std::cout << "!";
    } else {
      if (c == ' ') {
        if (letter > 0) {
          //std::cout << " " << letter << " ";
          localWordCount++;
          letter = 0;
          //std::cout << "*";
        }
      }
    }
    
  }
  if (letter > 0) {
    //std::cout << " " << letter << " ";
    localWordCount++;
    letter = 0;
    //std::cout << "&";
  }
  //std::cout << " " << localWordCount << std::endl;
  
  boost::mpi::reduce(world, localWordCount, wordCount, std::plus<>(), 0);
  if (world.rank() == 0) {
    wordCount -= fakeWord;
    std::cout << "HERE IS FAKE WORD: " << fakeWord << " AND HERE IS WORD COUNT: " << wordCount << std::endl;
  }

    std::cout << localWordCount << " " << world.rank() << std::endl;
  return true;
}

bool chernova_n_word_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  }
  return true;
}
