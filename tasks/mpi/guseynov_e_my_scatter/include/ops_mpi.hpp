#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"

namespace guseynov_e_my_scatter_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res_{};
};

class MyScatterTestMPITaskParallel : public ppc::core::Task {
 public:
  explicit MyScatterTestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  template <typename T>
  static void my_scatter(const boost::mpi::communicator& comm, const std::vector<T>& in_values, T* out_values, int n,
                         int root) {
    int left_child = 2 * comm.rank() + 1;
    int right_child = 2 * comm.rank() + 2;
    int level = static_cast<int>(std::floor(log(comm.rank() + 1) / std::numbers::ln2));
    int proc_on_level = static_cast<int>(pow(2, level));
    if (comm.rank() == root) {
      std::copy(in_values.begin(), in_values.begin() + n, out_values);
      if (left_child < comm.size()) {
        comm.send(left_child, 0, in_values.data() + n, (comm.size() - 1) * n);
      }
      if (right_child < comm.size()) {
        comm.send(right_child, 0, in_values.data() + n, (comm.size() - 1) * n);
      }
    } else {
      int min_rank_on_level = proc_on_level - 1;
      int recv_buffer_size = (comm.size() - min_rank_on_level) * n;
      int recv_id = (comm.rank() - 1) / 2;
      std::vector<T> recv_buffer(recv_buffer_size);

      comm.recv(recv_id, 0, recv_buffer.data(), recv_buffer_size);
      std::copy(recv_buffer.begin() + (comm.rank() - min_rank_on_level) * n,
                recv_buffer.begin() + (comm.rank() - min_rank_on_level) * n + n, out_values);
      if (left_child < comm.size()) {
        comm.send(left_child, 0, recv_buffer.data() + n * proc_on_level, (comm.size() - min_rank_on_level * 2 - 1) * n);
      }
      if (right_child < comm.size()) {
        comm.send(right_child, 0, recv_buffer.data() + n * proc_on_level,
                  (comm.size() - min_rank_on_level * 2 - 1) * n);
      }
    }
  }

 private:
  std::vector<int> input_, local_input_;
  int res_{};
  boost::mpi::communicator world;
};
}  // namespace guseynov_e_my_scatter_mpi