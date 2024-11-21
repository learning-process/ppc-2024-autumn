#include "mpi/dormidontov_e_min_value_by_columns_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  rs = taskData->inputs_count[0];
  cs = taskData->inputs_count[1];

  input_.resize(rs, std::vector<int>(cs));

  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < cs; ++j) {
      input_[i][j] = reinterpret_cast<int*>(taskData->inputs[0])[i * (cs) + j];
    }
  }

  res_.resize(cs, 0);
  return true;
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return ((taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int j = 0; j < cs; ++j) {
    res_[j] = INT_MAX;
    for (int i = 0; i < rs; ++i) {
      if (res_[j] > input_[i][j]) {
        res_[j] = input_[i][j];
      }
    }
  }
  return true;
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cs; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    rs = taskData->inputs_count[0];
    cs = taskData->inputs_count[1];
    input_.resize(rs * cs);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + rs * cs,
              input_.begin());
  }
  return true;
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (((taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
             taskData->inputs_count[1] == taskData->outputs_count[0]));
  }
  return true;
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    rs = taskData->inputs_count[0];
    cs = taskData->inputs_count[1];
  }
  boost::mpi::broadcast(world, rs, 0);
  boost::mpi::broadcast(world, cs, 0);

  int unfitrs = rs % world.size();
  int rsperpro = rs / world.size();
  int locrs;
  int prs;
  int a;
  if (world.rank() < unfitrs) {
    locrs = rsperpro + 1;
  } else {
    locrs = rsperpro;
  }
  std::vector<int> locmin(cs, INT_MAX);

  minput_.resize(cs * locrs);

  if (world.rank() == 0) {
    a = locrs * cs;
    for (int i = 1; i < world.size(); i++) {
      if (i < unfitrs) {
        prs = rsperpro + 1;
      } else {
        prs = rsperpro;
      }
      world.send(i, 2, input_.data() + a, prs * cs);
      a += cs * prs;
    }
    std::copy(input_.begin(), input_.begin() + locrs * cs, minput_.begin());
  } else {
    world.recv(0, 2, minput_.data(), locrs * cs);
  }

  for (int i = 0; i < locrs; ++i) {
    for (int j = 0; j < cs; ++j) {
      if (locmin[j] > minput_[i * cs + j]) {
        locmin[j] = minput_[i * cs + j];
      }
    }
  }
  res_.resize(cs, 0);
  boost::mpi::reduce(world, locmin.data(), cs, res_.data(), boost::mpi::minimum<int>(), 0);
  return true;
}

bool dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < cs; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
    }
  }
  return true;
}