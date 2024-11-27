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

namespace mironov_a_broadcast_custom_mpi {

class ComponentSumPowerCustomImpl : public ppc::core::Task {
 public:
  explicit ComponentSumPowerCustomImpl(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  template <typename T>
  static void broadcastImpl(const boost::mpi::communicator& comm, T& value, int root)
  {
    if (comm.size() == 1) {
      return;
    }
    if (comm.size() == 2) {
      if (comm.rank() == root) {
        comm.send(1 - root, 0, value);
      } else {
        comm.recv(root, 0, value);
      }
      return;
    }

    if (comm.rank() == root) {
      comm.send((root + 1) % comm.size(), 0, value);
      comm.send((root + 2) % comm.size(), 0, value);
    } else {
      int id_elem = comm.rank() - root;
      if (comm.rank() < root) {
        id_elem = comm.size() - root + comm.rank();
      }
      int id_sender = (root + (id_elem - 1) / 2) % comm.size();
      int id1 = 2 * id_elem + 1;
      int id2 = 2 * id_elem + 2;

      comm.recv(id_sender, 0, value);

      if (id1 < comm.size()) {
        comm.send((root + id1) % comm.size(), 0, value);
      }
      if (id2 < comm.size()) {
        comm.send((root + id2) % comm.size(), 0, value);
      }
    }
  }

  template <typename T>
  static void broadcastImpl(const boost::mpi::communicator& comm, T* values, int n, int root)
  {
    if (comm.size() == 1) {
      return;
    }
    if (comm.size() == 2) {
      if (comm.rank() == root) {
        comm.send(1 - root, 0, values, n);
      } else {
        comm.recv(root, 0, values, n);
      }
      return;
    }

    if (comm.rank() == root) {
      comm.send((root + 1) % comm.size(), 0, values, n);
      comm.send((root + 2) % comm.size(), 0, values, n);
    } else {
      int id_elem = comm.rank() - root;
      if (comm.rank() < root) {
        id_elem = comm.size() - root + comm.rank();
      }
      int id_sender = (root + (id_elem - 1) / 2) % comm.size();
      int id1 = 2 * id_elem + 1;
      int id2 = 2 * id_elem + 2;

      comm.recv(id_sender, 0, values, n);

      if (id1 < comm.size()) {
        comm.send((root + id1) % comm.size(), 0, values, n);
      }
      if (id2 < comm.size()) {
        comm.send((root + id2) % comm.size(), 0, values, n);
      }
    }
  }

 private:
  std::vector<int> input_;
  std::vector<int> powers_;
  std::vector<int> result_;
  int delta_ = 0;
  int size_ = 0;
  boost::mpi::communicator world;
};

}  // namespace mironov_a_broadcast_custom_mpi
