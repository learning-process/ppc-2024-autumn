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

namespace sidorina_p_broadcast_mpi {

class Broadcast : public ppc::core::Task {
 public:
  explicit Broadcast(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  template <typename T>
  static void broadcast_m(const boost::mpi::communicator& comm, T& value, int root) {
    broadcast_m(comm, &value, 1, root);
  }

  template <typename T>
  static void broadcast_m(const boost::mpi::communicator& comm, T* value, int n, int root) {
    if (comm.size() == 1) {
      return;
    }
    if (comm.size() <= 3) {
      if (comm.rank() == root) {
        for (int i = 0; i < root; i++) {
          comm.send(i, 0, value, n);
        }
        for (int i = root + 1; i < comm.size(); i++) {
          comm.send(i, 0, value, n);
        }
      } else {
        comm.recv(root, 0, value, n);
      }
      return;
    }

    if (comm.rank() == root) {
      for (int j = 1; j < 3; j++) {
        comm.send((root + j) % comm.size(), 0, value, n);
      }
    } else {
      int id_el = comm.rank() - root;
      if (comm.rank() < root) {
        id_el = comm.size() - root + comm.rank();
      }
      int id_send = (root + (id_el - 1) / 2) % comm.size();
      comm.recv(id_send, 0, value, n);
      for (int i = 1; i < 3; i++) {
        if ((2 * id_el + i) < comm.size()) {
          comm.send((root + 2 * id_el + i) % comm.size(), 0, value, n);
        }
      }
    }
  }

  std::function<void(const boost::mpi::communicator&, int*, int, int)> broadcast_fn;

 private:
  std::vector<int> arr;
  std::vector<int> term;
  std::vector<int> res;
  int del = 0;
  int sz = 0;
  boost::mpi::communicator world;
};
}  // namespace sidorina_p_broadcast_mpi