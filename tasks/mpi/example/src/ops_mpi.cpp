#include "mpi/example/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<char> volochaev_s_count_characters_27_mpi::get_random_string(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<char> vec(sz);
  for (size_t i = 0; i < sz; i++)
  {
    vec[i] += gen() % 256;
  }
  return vec;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::pre_processing() {
  internal_order_test();
  // Init vectors
  input1_ = std::vector<char>(taskData->inputs_count[0]);
  input2_ = std::vector<char>(taskData->inputs_count[1]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) 
  {
    input1_[i] = tmp_ptr[i];
  }

  tmp_ptr = reinterpret_cast<char*>(taskData->inputs[1]);
  for (size_t i = 0; i < taskData->inputs_count[1]; i++)
  {
    input2_[i] = tmp_ptr[i];
  }

  // Init value for output
  res = 0;
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::run() {
  internal_order_test();
  
  res = abs((int)input1_.size() - (int)input2_.size());

  for (size_t i = 0; i < std::min(input1_.size(), input2_.size()); ++i)
  {
     if (input1_[i] != input2_[i])
     {
       res += 2;
     }
  }

  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  if (world.rank() == 0) 
  {
     delta = (std::max(taskData->inputs_count[0] - 1, taskData->inputs_count[1] - 1)) / world.size();
     if (std::max(taskData->inputs_count[0] - 1, taskData->inputs_count[1] - 1) % world.size() > 0u) ++delta;
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) 
  {
    // Init vectors
    input_ = std::vector<std::pair<char, char>>(world.size() * delta);
    auto tmp1_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    auto tmp2_ptr = reinterpret_cast<char*>(taskData->inputs[1]);
    for (unsigned i = 0; i < std::min(taskData->inputs_count[0], taskData->inputs_count[1]); i++) {
      input_[i].first = tmp1_ptr[i];
      input_[i].second = tmp2_ptr[i];
    }

    for (size_t i = std::min(taskData->inputs_count[0], taskData->inputs_count[1]);
         i < std::max(taskData->inputs_count[0], taskData->inputs_count[1]); i++)
    {
      if (taskData->inputs_count[0] > taskData->inputs_count[1])
      {
        input_[i].first = tmp1_ptr[i];
        input_[i].second = -255;
      }
      else
      {
        input_[i].first = -255;
        input_[i].second = tmp2_ptr[i];
      }
    }

    for (size_t proc = 1; proc < world.size(); proc++) 
    {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }

  local_input_ = std::vector<std::pair<char,char>>(delta);
  if (world.rank() == 0) 
  {
    local_input_ = std::vector<std::pair<char,char>>(input_.begin(), input_.begin() + delta);
  }
  else
  {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = 0;
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::validation() {
  internal_order_test();
  if (world.rank() == 0)
  {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::run() {
  internal_order_test();
  int local_res = 0;
  
  for (size_t i = 0; i < local_input_.size(); ++i)
  {
    if (local_input_[i].first != local_input_[i].second) local_res += 2;
  }

  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0) 
  {
    *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  } 
  return true;
}
