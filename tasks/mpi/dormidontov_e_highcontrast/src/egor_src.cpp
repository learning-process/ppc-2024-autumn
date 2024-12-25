#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/dormidontov_e_highcontrast/include/egor_include.hpp"

namespace dormidontov_e_highcontrast_mpi {
// use to control diap of data
void clip(int& x) {
  if (x > 255) x = 255;
  if (x < 0) x = 0;
}
}  // namespace dormidontov_e_highcontrast_mpi

bool dormidontov_e_highcontrast_mpi::ContrastS::pre_processing() {
  internal_order_test();
  size = taskData->inputs_count[0];
  y.resize(size);
  res_.resize(size);
  auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(ptr, ptr + size, y.data());
  return true;
}

bool dormidontov_e_highcontrast_mpi::ContrastS::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 1 && taskData->inputs_count[0] > 0) &&
         (taskData->outputs.size() == 1 && taskData->outputs_count[0] > 0);
}

bool dormidontov_e_highcontrast_mpi::ContrastS::run() {
  internal_order_test();
  ymin = 255;
  ymax = 0;
  for (int i = 0; i < size; ++i) {
    ymin = std::min(y[i], ymin);
    ymax = std::max(y[i], ymax);
  }
  clip(ymin);
  clip(ymax);
  for (int i = 0; i < size; ++i) {
    res_[i] = ((y[i] - ymin) * 255) / (ymax - ymin);
    clip(res_[i]);
  }
  return true;
}

bool dormidontov_e_highcontrast_mpi::ContrastS::post_processing() {
  internal_order_test();
  auto* tmp = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.data(), res_.data() + size, tmp);
  return true;
}

bool dormidontov_e_highcontrast_mpi::ContrastP::pre_processing() {
  internal_order_test();
  return true;
}

bool dormidontov_e_highcontrast_mpi::ContrastP::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs.size() == 1 && taskData->inputs_count[0] > 0) &&
           (taskData->outputs.size() == 1 && taskData->outputs_count[0] > 0);
  }
  return true;
}

bool dormidontov_e_highcontrast_mpi::ContrastP::run() {
  internal_order_test();

  int delta;
  int rest;
  if (world.rank() == 0) {
    size = taskData->inputs_count[0];
    delta = size / world.size();
    rest = size % world.size();
    Y.resize(size);
    auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(ptr, ptr + size, Y.data());
  }
  boost::mpi::broadcast(world, size, 0);
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, rest, 0);

  int data_size;
  if (world.rank() < rest) {
    data_size = delta + 1;
  } else {
    data_size = delta;
  }
  int Ymax = 0;
  int Ymin = 255;
  y.resize(data_size);

  int pos;
  if (world.rank() == 0) {
    int start = data_size;
    int ds;
    for (int i = 1; i < world.size(); i++) {
      if (i < rest) {
        ds = delta + 1;
      } else {
        ds = delta;
      }
      world.send(i, 0, Y.data() + start, ds);
      world.send(i, 0, start);
      start += ds;
    }
    std::copy(Y.begin(), Y.begin() + data_size, y.begin());
    pos = 0;
  } else {
    world.recv(0, 0, y.data(), data_size);
    world.recv(0, 0, pos);
  }
  for (int i = 0; i < data_size; ++i) {
    Ymin = std::min(y[i], Ymin);
    Ymax = std::max(y[i], Ymax);
  }

  boost::mpi::all_reduce(world, Ymin, ymin, boost::mpi::minimum<int>());
  boost::mpi::all_reduce(world, Ymax, ymax, boost::mpi::maximum<int>());

  std::vector<int> local_data_(size);
  for (int i = 0; i < data_size; ++i) {
    local_data_[pos + i] = ((y[i] - ymin) * 255) / (ymax - ymin);
    clip(local_data_[pos + i]);
  }
  res_.resize(size);
  boost::mpi::reduce(world, local_data_.data(), size, res_.data(), std::plus<>(), 0);
  return true;
}

bool dormidontov_e_highcontrast_mpi::ContrastP::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.data(), res_.data() + size, tmp);
  }
  return true;
}