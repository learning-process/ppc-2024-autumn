#pragma once

#include <boost/mpi/communicator.hpp>

#include "mpi/ermolaev_v_allreduce/include/shared_ptr_array.hpp"

namespace ermolaev_v_allreduce_mpi {

template <typename _T, typename _Op>
void allreduce(const boost::mpi::communicator& world, const _T* src, int n, _T* out, _Op op) {
  // init
  int size = world.size();
  int rank = world.rank();

  // k-ary Tree (with hardcoded optimal k)
  int k = 3;
  int grandparent = 0;
  int parent = 0;
  std::vector<int> children(k, -1);

  int state = rank;
  int multiplier = 1;
  while (state > 0) {
    grandparent = parent;
    parent += (state % k) * multiplier;
    state /= k;
    multiplier *= k;
  }

  for (int i = 0; i < k; i++) {
    auto value = parent + (i + 1) * multiplier;
    if (value < size) children[i] = value;
  }

  // reduce part
  if (rank == 0) {
    std::copy(src, src + n, out);

    auto incoming_buffer = ermolaev_v_allreduce_mpi::shared_ptr_array<_T>(n);

    for (int i = 0; i < k; i++) {
      if (children[i] != -1) {
        world.recv(children[i], 0, incoming_buffer.get(), n);

        for (int j = 0; j < n; j++) {
          out[j] = op(out[j], incoming_buffer[j]);
        }
      }
    }
  } else {
    world.send(grandparent, 0, src, n);
  }

  // broadcast part
  if (rank != 0) world.recv(grandparent, 0, out, n);

  for (int i = 0; i < k; i++)
    if (children[i] != -1) world.send(children[i], 0, out, n);
}

}  // namespace ermolaev_v_allreduce_mpi