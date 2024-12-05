#include "mpi/shpynov_n_mismatched_characters_amount/include/mismatched_numbers_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define shortest_string_size (std::min(input_[0].size(), input_[1].size()))

using namespace std::chrono_literals;

int positive_min(int a, int b) {
  return std::min(a, b) < 0 ? 0 : std::min(a, b);
}  // returns std::min(a,b) or 0 if it's negative

int unique_characters(std::vector<std::string> const &vec1) {  // count unique characters
  std::string s1 = vec1[0];
  std::string s2 = vec1[1];
  int diff = abs(int(s1.size() - s2.size()));
  int count = diff;
  for (unsigned int i = 0; i < std::min(s1.size(), s2.size()); i++) {
    if (s1[i] != s2[i]) count++;
  }
  return count;
}

// *** SEQUENTIAL ALGORITHM *** //

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();  // recieving data
  input_.push_back(std::string(reinterpret_cast<char *>(taskData->inputs[0])));
  input_.push_back(std::string(reinterpret_cast<char *>(taskData->inputs[1])));

  res = 0;

  return true;
}

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential::validation() {
  internal_order_test();  // control amount of inputs/outputs
  return ((taskData->inputs_count[0] == 2) && (taskData->outputs_count[0] == 1));
}

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = unique_characters(input_);  // calculating result
  return true;
}

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}

// *** PARALLEL ALGORITHM *** //

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {  // root receiving input data
    res = 0;
    input_.push_back(std::string(reinterpret_cast<char *>(taskData->inputs[0])));
    input_.push_back(std::string(reinterpret_cast<char *>(taskData->inputs[1])));
  }
  return true;
}

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return ((taskData->inputs_count[0] == 2) && (taskData->outputs_count[0] == 1));
  }
  return true;
}

bool shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int delta = 0;
  if (world.rank() == 0) {  // root calculating delta
    if (shortest_string_size % world.size() == 0)
      delta = shortest_string_size / world.size();
    else
      delta = (shortest_string_size / world.size()) +
              1;  // if shortest strings' length is not multiple of number of
                  // processes, delta should be bigger to completely cover both strings
  }

  broadcast(world, delta, 0);
  if (world.rank() == 0) {  // root sending data to processes
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_[0].data() + proc * delta,
                 positive_min(delta, (int)shortest_string_size - proc * delta));
      // if last process is about to go OoB, shorter string should be assigned to it
      // (only down to 0 if amount of processes is greater than shortest string's length)
      world.send(proc, 0, input_[1].data() + proc * delta,
                 positive_min(delta, (int)shortest_string_size - proc * delta));
    }
  }
  if (world.rank() == 0) {  // root receiving its data
    local_input_.push_back(input_[0].substr(0, delta));
    local_input_.push_back(input_[1].substr(0, delta));
  } else {  // processes receiving their data
    std::string tmp1, tmp2;
    tmp1.resize(delta);
    tmp2.resize(delta);

    world.recv(0, 0, tmp1.data(), delta);
    world.recv(0, 0, tmp2.data(), delta);

    local_input_.push_back(tmp1);
    local_input_.push_back(tmp2);
  }

  int loc_res = unique_characters(local_input_);  // counting unique characters
  reduce(world, loc_res, res, std::plus(), 0);
  return true;
}
bool shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] =
        res + abs((int)std::string(reinterpret_cast<char *>(taskData->inputs[0])).size() -
                  (int)std::string(reinterpret_cast<char *>(taskData->inputs[1]))
                      .size());  // res is amount of unique characters up to the shortest string's length, every
                                 // character left in greater string should be also added to the result
  }
  return true;
}