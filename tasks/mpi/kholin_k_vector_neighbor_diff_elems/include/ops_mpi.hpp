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

namespace kholin_k_vector_neighbor_diff_elems_mpi {

template <class TypeElem, class TypeIndex>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
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
  std::string ops;
  TypeElem convert(const auto& elem);
};

template <class TypeElem, class TypeIndex>
bool TestTaskSequential<TypeElem, TypeIndex>::pre_processing() {
  internal_order_test();
  // Data TaskData  cite to type elements of vector input_
  input_ = std::vector<TypeElem>(taskData->inputs_count[0]);
  auto ptr = reinterpret_cast<TypeElem*>(taskData->inputs[0]);
  std::copy(ptr, ptr + taskData->inputs_count[0], std::back_inserter(input_));
  // Execute the actions as if this were the default constructor
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
  // Check count elements of output
  return taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2;
}

template <class TypeElem, class TypeIndex>
bool TestTaskSequential<TypeElem, TypeIndex>::run() {
  internal_order_test();
  // here your algorithm task (.h files for task or all in run)
  if (ops == "MAX_DIFFERENCE") {  // declaration operations with
    // start delta between elements vector
    double max_delta = 0;
    double delta = 0;
    size_t curr_index = 0;
    // get iterator for current element and his neighbor element vector
    auto iter_curr = input_.begin();
    auto iter_next = iter_curr + 1;
    auto iter_end = input_.end() - 1;
    auto iter_begin = input_.begin();
    // algorithm search max delta with using address arithmetic pointers
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
    // initialize results
    result = max_delta;
    // std::cout << result; //max delta here
    right_index = curr_index + 1;
    left_index = curr_index;
    left_elem = input_[left_index];
    right_elem = input_[right_index];
    // std::cout << "left el " << left_elem << "left_ind " << left_index << std::endl;
    // std::cout << "right el" << right_elem << "right_ind" << right_index << std::endl;
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
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}

  MPI_Datatype get_mpi_type();

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestMPITaskParallel() override { MPI_Type_free(&mpi_type_elem); }

 private:
  std::vector<TypeElem> input_;        // global vector
  std::vector<TypeElem> local_input_;  // local vector
  unsigned int delta_n;
  unsigned int delta_n_r;
  double result;
  unsigned int residue;
  std::string ops;
  boost::mpi::communicator world;
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
  // Data TaskData  cite to type elements of vector input_
  if (world.rank() == 0) {
    delta_n = taskData->inputs_count[0] / world.size();
    delta_n_r = {};
  }
  MPI_Bcast(&delta_n, 1, MPI_UNSIGNED, 0, world);  // send all procs delta_n
  if (world.rank() == 0) {
    input_ = std::vector<TypeElem>(taskData->inputs_count[0]);
    auto ptr = reinterpret_cast<TypeElem*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = ptr[i];
    }
    // distribute data processes 0 to size-1
  }
  if (world.rank() == 0) {
    residue = taskData->inputs_count[0] % world.size();
    delta_n_r = delta_n + residue;
    local_input_ = std::vector<TypeElem>(delta_n_r);
  } else {
    local_input_ = std::vector<TypeElem>(delta_n);
  }
  MPI_Scatter(input_.data(), delta_n, mpi_type_elem, local_input_.data(), delta_n, mpi_type_elem, 0, world);
  if (world.rank() == 0) {
    // write residue into vector process 0
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
  // Check count elements of output
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

template <typename TypeElem>
bool TestMPITaskParallel<TypeElem>::run() {
  internal_order_test();
  // output local_input_ vector
  double local_result = 0;
  if (ops == "MAX_DIFFERENCE") {
    // here your algorithm task (.h files for task or all in run)
    local_result = max_difference();
  }
  if (ops == "MAX_DIFFERENCE") {
    // everyone process send 1 element and get all local_results from everyone process 1 element
    double sendbuf1[1];
    sendbuf1[0] = local_result;
    MPI_Reduce(sendbuf1, &result, 1, MPI_DOUBLE, MPI_MAX, 0, world);
  }
  // finalisation
  return true;
}

template <typename TypeElem>
bool TestMPITaskParallel<TypeElem>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }
  return true;
}

template <typename TypeElem>
void TestMPITaskParallel<TypeElem>::print_local_data() {
  if (world.rank() == 0) {
    std::cout << "I'm proc 0" << "and my local_input data is ";
    for (unsigned int i = 0; i < delta_n_r; i++) {
      std::cout << local_input_[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "I'm" << world.rank() << " proc " << "and my local_input data is ";
    for (unsigned int i = 0; i < delta_n; i++) {
      std::cout << local_input_[i] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename TypeElem>
double TestMPITaskParallel<TypeElem>::max_difference() {
  // start delta between elements vector
  double max_delta = 0;
  double delta = 0;
  double local_result = 0;

  // get iterator for current element and his neighbor element vector
  auto iter_curr = local_input_.begin();
  auto iter_next = iter_curr + 1;
  auto iter_end = local_input_.end() - 1;
  // algorithm search max delta with using address arithmetic pointers
  while (iter_curr != iter_end) {
    delta = abs((double)(*iter_next - *iter_curr));
    if (delta > max_delta) {
      max_delta = delta;
    }
    iter_curr++;
    iter_next = iter_curr + 1;
    // initialize results
    local_result = max_delta;
  }
  return local_result;
}
}  // namespace kholin_k_vector_neighbor_diff_elems_mpi