#include "mpi/chistov_a_gather_boost/include/gather_boost.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace chistov_a_gather_boost {

template <typename T>
bool Gather<T>::pre_processing() {
  internal_order_test();

  int relativeRank = (world.rank() - root + world.size()) % world.size();
  parent = (world.rank() == root) ? -1 : (root + (relativeRank - 1) / 2) % world.size();
  int relativeLeftChild = 2 * relativeRank + 1;
  int relativeRightChild = 2 * relativeRank + 2;
  leftChild = (relativeLeftChild < world.size()) ? (root + relativeLeftChild) % world.size() : -1;
  rightChild = (relativeRightChild < world.size()) ? (root + relativeRightChild) % world.size() : -1;

  return true;
}

template <typename T>
bool Gather<T>::validation() {
  internal_order_test();

  if (root < 0 || root >= world.size()) {
    return false;
  }

  if (taskData->inputs_count.empty() || taskData->inputs_count[0] <= 0) {
    return false;
  }

  return true;
}

template <typename T>
bool Gather<T>::run() {
  internal_order_test();

  std::vector<T> localbuf(taskData->inputs_count[0]);
  memcpy(localbuf.data(), reinterpret_cast<T*>(taskData->inputs[0]), taskData->inputs_count[0] * sizeof(T));

  if (leftChild != -1) {
    int leftDataSize;
    world.recv(leftChild, 0, &leftDataSize, 1);

    std::vector<T> leftData(leftDataSize);
    world.recv(leftChild, 0, leftData.data(), leftDataSize);

    localbuf.insert(localbuf.end(), std::make_move_iterator(leftData.begin()), std::make_move_iterator(leftData.end()));
  }

  if (rightChild != -1) {
    int rightDataSize;
    world.recv(rightChild, 0, &rightDataSize, 1);

    std::vector<T> rightData(rightDataSize);
    world.recv(rightChild, 0, rightData.data(), rightDataSize);

    localbuf.insert(localbuf.end(), std::make_move_iterator(rightData.begin()),
                    std::make_move_iterator(rightData.end()));
  }

  if (world.rank() != root) {
    int localbufSize = localbuf.size();
    world.send(parent, 0, &localbufSize, 1);
    world.send(parent, 0, localbuf.data(), localbufSize);
  } else {
    sendbuf = std::move(localbuf);
  }

  return true;
}

template <typename T>
bool Gather<T>::post_processing() {
  internal_order_test();

  if (world.rank() == root) {
    std::memcpy(reinterpret_cast<T*>(taskData->outputs[0]), sendbuf.data(), sendbuf.size() * sizeof(T));
  }
  return true;
}

template class Gather<int>;
template class Gather<double>;
template class Gather<float>;

}  // namespace chistov_a_gather_boost