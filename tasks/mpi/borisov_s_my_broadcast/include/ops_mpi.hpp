// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_my_broadcast {

std::vector<double> getRandomPoints(int count);

class DistanceMatrixTaskSequential : public ppc::core::Task {
public:
  explicit DistanceMatrixTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

private:
  std::vector<double> points_;
  std::vector<double> distance_matrix_;
};

class DistanceMatrixTaskParallel : public ppc::core::Task {
public:
  explicit DistanceMatrixTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

private:
  std::vector<double> points_;
  std::vector<double> local_points_;
  std::vector<double> distance_matrix_;
  boost::mpi::communicator world;
};


template <typename T>
void my_broadcast(const boost::mpi::communicator &comm, T &value, int root) {
  int size = comm.size();
  int rank = comm.rank();

  if (size == 1) return;

  int source = (rank - root + size) % size;
  int parent;
  if (rank == root) {
    parent = -1;
  } else {
    parent = (root + (source - 1) / 3) % size;
  }

  if (rank != root) {
    comm.recv(parent, 0, value);
  }

  for (int i = 1; i <= 3; i++) {
    int child = root + (3 * source) + i;
    if (child < size) {
      comm.send(child, 0, value);
    }
  }
}


template <typename T>
void my_broadcast(const boost::mpi::communicator &comm, T *values, int n, int root) {
  int size = comm.size();
  int rank = comm.rank();

  if (size == 1) return;

  int source = (rank - root + size) % size;
  int parent;
  if (rank == root) {
    parent = -1;
  } else {
    parent = (root + (source - 1) / 3) % size;
  }

  if (rank != root) {
    comm.recv(parent, 0, values, n);
  }

  for (int i = 1; i <= 3; ++i) {
    int child = root + (3 * source) + i;
    if (child < size) {
      comm.send(child, 0, values, n);
    }
  }
}

}  // namespace borisov_s_my_broadcast