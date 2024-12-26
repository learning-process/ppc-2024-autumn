#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorochkin_d_test_task_mpi {

template <typename T>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override {
    internal_order_test();

    if (world.rank() == 0) {
      const auto* src = reinterpret_cast<T*>(taskData->inputs[0]);
      const auto len = taskData->inputs_count[0];
      buf.assign(src, src + len);
    }

    return true;
  }

  bool validation() override {
    internal_order_test();
    return world.rank() != 0 || (taskData->inputs.size() == 1 && taskData->outputs.size() == 1 &&
                                 taskData->inputs_count[0] == taskData->outputs_count[0]);
  }

  bool run() override {
    internal_order_test();

    const auto rank = world.rank();
    const auto np = world.size();

    if (np == 1) {
      return true;
    }

    if (rank == 0) {
      do_send(rank + 1);
    }
    do_recv(rank == 0 ? (np - 1) : (rank - 1));
    if (rank != 0) {
      do_send(rank == (np - 1) ? 0 : rank + 1);
    }

    return true;
  }

  bool post_processing() override {
    internal_order_test();
    if (world.rank() == 0) {
      auto* dst = reinterpret_cast<T*>(taskData->outputs[0]);
      std::copy(buf.begin(), buf.end(), dst);
    }
    return true;
  }

 private:
  void do_send(int dst) {
    const auto size = buf.size();
    world.send(dst, 0, size);
    world.send(dst, 0, buf.data(), size);
  }
  void do_recv(int src) {
    size_t size;
    world.recv(src, 0, size);
    buf.resize(size);
    world.recv(src, 0, buf.data(), size);
  }

  std::vector<T> buf;
  boost::mpi::communicator world;
};

}  // namespace sorochkin_d_test_task_mpi