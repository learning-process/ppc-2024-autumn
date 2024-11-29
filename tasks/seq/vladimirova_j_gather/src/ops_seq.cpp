// Copyright 2024 Nesterov Alexander
#include "seq/vladimirova_j_gather/include/ops_seq.hpp"

#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> vladimirova_j_gather_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = (gen() % 2 + 1) * (gen() % 2 - 1);
  }
  return vec;
}

std::vector<int> vladimirova_j_gather_seq::noDeadEnds(std::vector<int> way) {
  int i = 0;
  size_t j = 1;

  while (j <= way.size()) {
    if ((way[i] * way[i] == 1) && (way[i] == way[j])) {
      do {
        i -= 1;
        j += 1;
        if (!(((size_t)i >= 0) && (j < way.size()))) break;
        if (((way[i] * way[i] == 1) && (way[i] == (-1) * way[j])) ||
            (way[i] * way[j] == 4)) {  // if rl lr or uu dd   1-1 -11 or 22 -2-2
          way[i] = 0;
          way[j] = 0;

        } else {
          break;
        }

      } while ((i > 0) && (j < way.size()));
      i = j - 1;
    }

    j++;
    i++;
  }

  way.erase(std::remove(way.begin(), way.end(), 0), way.end());
  return way;
}

std::vector<int> vladimirova_j_gather_seq::noStrangeSteps(std::vector<int> way) {
  std::cout << "NO STR STEPS\n";
  for (int value : way) {
    std::cout << value << " ";
  }

  for (size_t i = 1; i < way.size(); i++)
    if (way[i] == -way[i - 1]) {
      way[i] = 0;
      way[i - 1] = 0;
    }
  way.erase(std::remove(way.begin(), way.end(), 0), way.end());

  for (size_t i = 3; i < way.size(); i++) {
    if (((way[i] == -1) || (way[i] == 1)) && (way[i] == way[i - 1]) && (way[i] == way[i - 2]) &&
        (way[i] == way[i - 3])) {
      way[i] = 0;
      way[i - 1] = 0;
      way[i - 2] = 0;
      way[i - 3] = 0;
    }
  }

  way.erase(std::remove(way.begin(), way.end(), 0), way.end());
  std::cout << "NO STR STEPS\n";
  for (int value : way) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  return way;
}

bool vladimirova_j_gather_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  return true;
}

bool vladimirova_j_gather_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool vladimirova_j_gather_seq::TestTaskSequential::run() {
  internal_order_test();
  input_ = vladimirova_j_gather_seq::noDeadEnds(input_);
  res = vladimirova_j_gather_seq::noStrangeSteps(input_);
  return true;
}

bool vladimirova_j_gather_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  taskData->outputs_count[0] = res.size();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), output_data);
  // reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}
