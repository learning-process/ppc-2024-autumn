#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_m_broadcast_mpi {

template <typename T>
void bcast(const boost::mpi::communicator &comm, T &value, int root) {
  comm.barrier();
  int rank = comm.rank();
  int size = comm.size();

  int relative_rank = (rank + size - root) % size;
  int mask = 1;

  while (mask < size) {
    if (relative_rank < mask) {
      int dst = relative_rank + mask;
      if (dst < size) {
        int dst_rank = (dst + root) % size;
        comm.send(dst_rank, 0, value);
      }
    } else if (relative_rank < 2 * mask) {
      int src = relative_rank - mask;
      int src_rank = (src + root) % size;
      comm.recv(src_rank, 0, value);
    }
    mask <<= 1;
  }
}

class MyBroadcastParallelMPI : public ppc::core::Task {
 public:
  explicit MyBroadcastParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int source_worker;
  std::vector<int> A;
  int global_sum_A;
  boost::mpi::communicator world;
};

class MpiBroadcastParallelMPI : public ppc::core::Task {
 public:
  explicit MpiBroadcastParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int source_worker;
  std::vector<int> A;
  int global_sum_A;
  boost::mpi::communicator world;
};

}  // namespace muradov_m_broadcast_mpi
