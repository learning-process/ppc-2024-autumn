#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace enum_ops {
enum operations { MAX_DIFFERENCE };
};

namespace kholin_k_vector_neighbor_diff_elems_mpi {

template <class TypeElem>
std::vector<TypeElem> get_random_vector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<TypeElem> vec(sz);

  if (std::is_integral<TypeElem>::value) {
    std::uniform_int_distribution<int> dist(0, 99);
    for (int i = 0; i < sz; i++) {
      vec[i] = dist(gen);
    }
  } else if (std::is_floating_point<TypeElem>::value) {
    std::uniform_real_distribution<float> dist(0, 99);
    for (int i = 0; i < sz; i++) {
      vec[i] = dist(gen);
    }
  } else {
    throw std::invalid_argument("TypeElem must be an integral or floating point type");
  }

  return vec;
}

template <class TypeElem, class TypeIndex>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, enum_ops::operations ops_)
      : Task(std::move(taskData_)), ops(ops_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<TypeElem> input_;
  double result;
  TypeIndex left_index;
  TypeIndex right_index;
  TypeElem left_elem;
  TypeElem right_elem;
  enum_ops::operations ops;
};

template <class TypeElem, class TypeIndex>
bool TestTaskSequential<TypeElem, TypeIndex>::pre_processing() {
  internal_order_test();
  input_ = std::vector<TypeElem>(taskData->inputs_count[0]);
  auto ptr = reinterpret_cast<TypeElem*>(taskData->inputs[0]);
  std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
  result = {};
  left_index = {};
  right_index = 2;
  left_elem = {};
  right_elem = {};
  return true;
}

template <class TypeElem, class TypeIndex>
bool TestTaskSequential<TypeElem, TypeIndex>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2;
}

template <class TypeElem, class TypeIndex>
bool TestTaskSequential<TypeElem, TypeIndex>::run() {
  internal_order_test();
  if (ops == enum_ops::MAX_DIFFERENCE) {
    double max_delta = 0;
    double delta = 0;
    size_t curr_index = 0;
    auto iter_curr = input_.begin();
    auto iter_next = iter_curr + 1;
    auto iter_end = input_.end() - 1;
    auto iter_begin = input_.begin();
    while (iter_curr != iter_end) {
      delta = abs(*iter_next - *iter_curr);
      if (delta > max_delta) {
        if (iter_begin == iter_curr) {
          curr_index = 0;
          max_delta = delta;
        } else {
          curr_index = std::distance(input_.begin(), iter_curr);
          max_delta = delta;
        }
      }
      iter_curr++;
      iter_next = iter_curr + 1;
    }
    result = max_delta;
    right_index = curr_index + 1;
    left_index = curr_index;
    left_elem = input_[left_index];
    right_elem = input_[right_index];
  }
  return true;
}

template <class TypeElem, class TypeIndex>
bool TestTaskSequential<TypeElem, TypeIndex>::post_processing() {
  internal_order_test();
  reinterpret_cast<TypeElem*>(taskData->outputs[0])[0] = left_elem;
  reinterpret_cast<TypeElem*>(taskData->outputs[0])[1] = right_elem;
  reinterpret_cast<TypeIndex*>(taskData->outputs[1])[0] = left_index;
  reinterpret_cast<TypeIndex*>(taskData->outputs[1])[1] = right_index;
  reinterpret_cast<double*>(taskData->outputs[2])[0] = result;
  return true;
}

template <class TypeElem>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, enum_ops::operations ops_)
      : Task(std::move(taskData_)), ops(ops_) {}

  MPI_Datatype get_mpi_type();

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestMPITaskParallel() override { MPI_Type_free(&mpi_type_elem); }

 private:
  std::vector<TypeElem> input_;
  std::vector<TypeElem> local_input_;
  unsigned int delta_n;
  unsigned int delta_n_r;
  double result;
  unsigned int residue;
  enum_ops::operations ops;
  MPI_Datatype mpi_type_elem;
  void print_local_data();
  double max_difference();
};

template <typename TypeElem>
MPI_Datatype TestMPITaskParallel<TypeElem>::get_mpi_type() {
  MPI_Type_contiguous(sizeof(TypeElem), MPI_BYTE, &mpi_type_elem);
  MPI_Type_commit(&mpi_type_elem);
  return mpi_type_elem;
}

template <typename TypeElem>
bool TestMPITaskParallel<TypeElem>::pre_processing() {
  internal_order_test();
  int ProcRank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (ProcRank == 0) {
    delta_n = taskData->inputs_count[0] / size;
    delta_n_r = {};
  }
  MPI_Bcast(&delta_n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (ProcRank == 0) {
    input_ = std::vector<TypeElem>(taskData->inputs_count[0]);
    auto ptr = reinterpret_cast<TypeElem*>(taskData->inputs[0]);
    std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
  }
  if (ProcRank == 0) {
    residue = taskData->inputs_count[0] % size;
    delta_n_r = delta_n + residue;
    local_input_ = std::vector<TypeElem>(delta_n_r);
  } else {
    local_input_ = std::vector<TypeElem>(delta_n);
  }
  MPI_Scatter(input_.data(), delta_n, mpi_type_elem, local_input_.data(), delta_n, mpi_type_elem, 0, MPI_COMM_WORLD);
  if (ProcRank == 0) {
    for (unsigned int i = delta_n; i < delta_n_r; i++) {
      local_input_[i] = input_[i];
    }
  }
  result = {};
  residue = {};
  return true;
}

template <typename TypeElem>
bool TestMPITaskParallel<TypeElem>::validation() {
  internal_order_test();
  mpi_type_elem = get_mpi_type();
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

template <typename TypeElem>
bool TestMPITaskParallel<TypeElem>::run() {
  internal_order_test();
  double local_result = 0;
  local_result = max_difference();
  if (ops == enum_ops::MAX_DIFFERENCE) {
    double sendbuf1[1];
    sendbuf1[0] = local_result;
    MPI_Reduce(sendbuf1, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }
  // finalisation
  return true;
}

template <typename TypeElem>
bool TestMPITaskParallel<TypeElem>::post_processing() {
  internal_order_test();
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }
  return true;
}

template <typename TypeElem>
void TestMPITaskParallel<TypeElem>::print_local_data() {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    std::cout << "I'm proc 0" << "and my local_input data is ";
    for (unsigned int i = 0; i < delta_n_r; i++) {
      std::cout << local_input_[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "I'm" << ProcRank << " proc " << "and my local_input data is ";
    for (unsigned int i = 0; i < delta_n; i++) {
      std::cout << local_input_[i] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename TypeElem>
double TestMPITaskParallel<TypeElem>::max_difference() {
  double max_delta = 0;
  double delta = 0;
  double local_result = 0;
  auto iter_curr = local_input_.begin();
  auto iter_next = iter_curr + 1;
  auto iter_end = local_input_.end() - 1;
  while (iter_curr != iter_end) {
    delta = abs(*iter_next - *iter_curr);
    if (delta > max_delta) {
      max_delta = delta;
    }
    iter_curr++;
    iter_next = iter_curr + 1;
  }
  local_result = max_delta;
  return local_result;
}
}  // namespace kholin_k_vector_neighbor_diff_elems_mpi