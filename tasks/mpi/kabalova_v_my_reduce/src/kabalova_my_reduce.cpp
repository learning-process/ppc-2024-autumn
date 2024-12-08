// Copyright 2024 Kabalova Valeria
#include "mpi/kabalova_v_my_reduce/include/kabalova_my_reduce.hpp"

#include <mpi.h>

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

// List of valid operations:
// + = MPI_SUM
// * = MPI_PROD
// min = MPI_MIN
// max = MPI_MAX
// && = MPI_LAND
// || = MPI_LOR
// & = MPI_BAND
// | = MPI_BOR
// ^ = MPI_BXOR
// lxor = MPI_LXOR

kabalova_v_my_reduce::Tree::Tree(int rank, int size, int root) {
  this->rank = rank;
  this->size = size;
  this->root = root;
  this->level_ = 0;

  int n = (rank + size - root) % size;
  int sum = 0;
  int count = 1;
  while (sum <= n) {
    level_++;
    count *= 2;
    sum += count;
  }
}

int kabalova_v_my_reduce::Tree::levelIndex(int n) {
  int sum = 0;
  int count = 1;
  // We go from the current layer to the parent up (parent has 0th layer)
  while ((n--) != 0) {
    sum += count;
    count *= 2;
  }
  return sum;
}

int kabalova_v_my_reduce::Tree::parent() const {
  if (rank == root) return root;
  int n = rank + size - 1 - root;
  int tmp = ((n % size / 2) + root) % size;
  return tmp;
}

int kabalova_v_my_reduce::Tree::begin() const {
  int n = (rank + size - root) % size;
  int childIndex = levelIndex(level_ + 1) + 2 * (n - levelIndex(level_));
  int result = (childIndex + root) % size;

  if (childIndex >= size) return root;
  return result;
}

bool kabalova_v_my_reduce::checkValidOperation(const std::string& ops) {
  return (ops == "+" || ops == "*" || ops == "min" || ops == "max" || ops == "&&" || ops == "||" || ops == "&" ||
          ops == "|" || ops == "^" || ops == "lxor");
}

int kabalova_v_my_reduce::op(const int& a, const int& b, const std::string& ops) {
  int tmp = 0;
  if (ops == "+") {
    tmp = a + b;
    if (tmp > std::numeric_limits<int>::max()) throw "overflow";
  } else if (ops == "*") {
    tmp = a * b;
    if (tmp > std::numeric_limits<int>::max()) throw "overflow";
  } else if (ops == "min") {
    tmp = std::min(a, b);
  } else if (ops == "max") {
    tmp = std::max(a, b);
  } else if (ops == "&&") {
    tmp = static_cast<int>(bool(a) && (bool)b);
  } else if (ops == "||") {
    tmp = static_cast<int>((bool)a || (bool)b);
  } else if (ops == "&") {
    tmp = a & b;
  } else if (ops == "|") {
    tmp = a | b;
  } else if (ops == "^") {
    tmp = a ^ b;
  } else if (ops == "lxor") {
    bool res1 = !(bool)a;
    bool res2 = !(bool)b;
    tmp = static_cast<int>(res1 != res2);
  }
  return tmp;
}

// Main function of reduce. Supports reducing at the root and for the root
void kabalova_v_my_reduce::myReduce(const boost::mpi::communicator& comm, const int& value, int& outValue,
                                    const std::string& ops, int root) {
  if (comm.rank() == root)
    kabalova_v_my_reduce::reduceImplementation(comm, value, outValue, ops, root);
  else
    kabalova_v_my_reduce::reduceImplementation(comm, value, ops, root);
}

// Reducing at the root with tree-based algorithm
void kabalova_v_my_reduce::reduceImplementation(const boost::mpi::communicator& comm, const int& inValue, int& outValue,
                                                const std::string& ops, int root) {
  kabalova_v_my_reduce::reduceTree(comm, inValue, outValue, ops, root);
}

// Reducing to the root with tree-based algorithm
void kabalova_v_my_reduce::reduceImplementation(const boost::mpi::communicator& comm, const int& inValue,
                                                const std::string& ops, int root) {
  kabalova_v_my_reduce::reduceTree(comm, inValue, ops, root);
}

// Commutative reduction
void kabalova_v_my_reduce::reduceTree(const boost::mpi::communicator& comm, const int& inValue, int& outValue,
                                      const std::string& ops, int root) {
  outValue = inValue;
  int size = comm.size();
  int rank = comm.rank();

  kabalova_v_my_reduce::Tree tree(rank, size, root);

  MPI_Status status;
  int children = 0;
  // begin() - returns the index for the first child of this process
  // We have binary tree so we go until 2
  // Child = (child + 1) % size - recalculate the next child of this process
  for (int child = tree.begin(); children < 2 && child != root; children++, child = (child + 1) % size) {
    // Receive archive
    boost::mpi::packed_iarchive iarchive(comm);
    boost::mpi::detail::packed_archive_recv(comm, child, 0, iarchive, status);
    int incoming;
    iarchive >> incoming;
    outValue = op(outValue, incoming, ops);
  }
  // For non-roots, send the result to the parent.
  if (tree.parent() != rank) {
    boost::mpi::packed_oarchive oarchive(comm);
    oarchive << outValue;
    boost::mpi::detail::packed_archive_send(comm, tree.parent(), 0, oarchive);
  }
}
// Commutative reduction from a non-root.
void kabalova_v_my_reduce::reduceTree(const boost::mpi::communicator& comm, const int& inValue, const std::string& ops,
                                      int root) {
  int result = 0;
  kabalova_v_my_reduce::reduceTree(comm, inValue, result, ops, root);
}

bool kabalova_v_my_reduce::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init value for input
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    local_input_ = std::vector<int>(taskData->inputs_count[0], 0);
  }
  return true;
}

bool kabalova_v_my_reduce::TestMPITaskParallel::validation() {
  internal_order_test();
  bool flag = true;
  if (world.rank() == 0) {
    flag = (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1 && checkValidOperation(ops));
  }
  broadcast(world, flag, 0);
  return flag;
}

bool kabalova_v_my_reduce::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int remains = 0;
  if (world.rank() == 0) {
    // Get delta = string.size() / num_threads
    delta = taskData->inputs_count[0] / world.size();
    // Because delta is not always an integer - this is what we got remained
    remains = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);
  broadcast(world, remains, 0);

  // Need proper arguments for scatterv
  std::vector<int> subvectorSizes(world.size(), delta);
  // Offsets for scatterv
  std::vector<int> offsets(world.size(), 0);
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < remains; i++) subvectorSizes[i]++;
    for (unsigned int i = 1; i < (unsigned int)world.size(); i++) offsets[i] = offsets[i - 1] + subvectorSizes[i - 1];

    // Local size of root process
    unsigned int localSize = subvectorSizes[0];
    local_input_.resize(localSize);

    boost::mpi::scatterv(world, input_.data(), subvectorSizes, offsets, local_input_.data(), localSize, 0);
  } else {
    if ((unsigned int)world.rank() < remains) {
      subvectorSizes[world.rank()]++;
    }
    // Local size for the current process
    unsigned int localSize = subvectorSizes[world.rank()];
    local_input_.resize(localSize);

    boost::mpi::scatterv(world, local_input_.data(), localSize, 0);
  }

  // After this we can finally use reduce
  int local_res;
  if (ops == "+") {  // MPI_SUM
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
    myReduce(world, local_res, result, "+", 0);
  } else if (ops == "*") {  // MPI_PROD
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res *= local_input_[i];
      if (local_res > std::numeric_limits<int>::max()) throw "overflow";
    }
    myReduce(world, local_res, result, "*", 0);
  } else if (ops == "max") {  // MPI_MAX
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
    myReduce(world, local_res, result, "max", 0);
  } else if (ops == "min") {  // MPI_MIN
    local_res = *std::min_element(local_input_.begin(), local_input_.end());
    myReduce(world, local_res, result, "min", 0);
  } else if (ops == "&&") {  // MPI_LAND
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = static_cast<int>((bool)local_res && (bool)local_input_[i]);
    }
    myReduce(world, local_res, result, "&&", 0);
  } else if (ops == "||") {  // MPI_LOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = static_cast<int>((bool)local_res || (bool)local_input_[i]);
    }
    myReduce(world, local_res, result, "||", 0);
  } else if (ops == "&") {  // MPI_BAND
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res & local_input_[i];
    }
    myReduce(world, local_res, result, "&", 0);
  } else if (ops == "|") {  // MPI_BOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res | local_input_[i];
    }
    myReduce(world, local_res, result, "|", 0);
  } else if (ops == "^") {  // MPI_BXOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res ^ local_input_[i];
    }
    myReduce(world, local_res, result, "^", 0);
  } else if (ops == "lxor") {  // MPI_LXOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      bool res1 = !(bool)local_res;
      bool res2 = !(bool)local_input_[i];
      local_res = static_cast<int>(res1 != res2);
    }
    myReduce(world, local_res, result, "lxor", 0);
  }
  return true;
}

bool kabalova_v_my_reduce::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}