// Copyright 2023 Liolya Seledkina
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace zaitsev_a_scatter {

int scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, MPI_Comm comm);

template <typename T, auto func>
  requires std::same_as<decltype(+func), int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm)>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, int root_, MPI_Datatype dtype_)
      : Task(std::move(taskData_)), root(root_), dtype(dtype_) {}

 private:
  std::vector<T> input;
  std::vector<T> local_input;
  int root;
  MPI_Datatype dtype;
  T res;
  boost::mpi::communicator world;

 public:
  bool pre_processing() override {
    internal_order_test();

    if (world.rank() == 0) {
      input = std::vector<T>(taskData->inputs_count[0]);
      auto* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input.begin());
    }
    res = 0;
    return true;
  };

  bool validation() override {
    internal_order_test();
    return world.size() > 1 && (world.rank() != 0 || taskData->inputs_count[0] > 0);
  };

  bool run() override {
    internal_order_test();

    unsigned int shift = 0;

    if (world.rank() == root) {
      shift = taskData->inputs_count[0] / world.size();
      local_input = std::vector<T>(shift + taskData->inputs_count[0] % world.size());
    }

    MPI_Bcast(&shift, 1, MPI_INT, root, world);

    if (world.rank() != root) local_input = std::vector<T>(shift);

    func(input.data(), shift, dtype, local_input.data(), shift, dtype, root, world);  // scatter

    if (world.rank() == root) {
      std::copy(input.begin() + shift * world.size(), input.end(), local_input.begin() + shift);
    }

    T local_res = 0;
    if (!local_input.empty()) {
      local_res = *std::min_element(local_input.begin(), local_input.end());
    }

    reduce(world, local_res, res, boost::mpi::minimum<T>(), root);

    return true;
  };
  bool post_processing() override {
    internal_order_test();

    if (world.rank() == 0) reinterpret_cast<T*>(taskData->outputs[0])[0] = res;

    return true;
  };
};

}  // namespace zaitsev_a_scatter