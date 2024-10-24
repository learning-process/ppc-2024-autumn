#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace kholin_k_vector_neighbor_diff_elems_mpi {

template <class TypeElem, class TypeIndex>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override {
    internal_order_test();
    // Data TaskData  cite to type elements of vector input_
    input_ = std::vector<TypeElem>(taskData->inputs_count[0]);
    auto ptr = reinterpret_cast<TypeElem*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = ptr[i];
    }
    // Execute the actions as if this were the default constructor
    result = {};
    left_index = {};
    right_index = 2;
    left_elem = {};
    right_elem = {};
    return true;
  }
  bool validation() override {
    internal_order_test();
    // Check count elements of output
    return taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2;
  }
  bool run() override {
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
  // get results
  bool post_processing() override {
    internal_order_test();
    reinterpret_cast<TypeElem*>(taskData->outputs[0])[0] = left_elem;
    reinterpret_cast<TypeElem*>(taskData->outputs[0])[1] = right_elem;
    reinterpret_cast<TypeIndex*>(taskData->outputs[1])[0] = left_index;
    reinterpret_cast<TypeIndex*>(taskData->outputs[1])[1] = right_index;
    return true;
  }

 private:
  std::vector<TypeElem> input_;
  double result;
  TypeIndex left_index;
  TypeIndex right_index;
  TypeElem left_elem;
  TypeElem right_elem;
  std::string ops;
};

template <class TypeElem, class TypeIndex>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}

  MPI_Datatype get_mpi_type() {
    MPI_Type_contiguous(sizeof(TypeElem), MPI_BYTE, &mpi_type_elem);
    MPI_Type_commit(&mpi_type_elem);
    return mpi_type_elem;
  }

   MPI_Datatype get_mpi_type2() {
    MPI_Type_contiguous(sizeof(TypeIndex), MPI_BYTE, &mpi_type_index);
    MPI_Type_commit(&mpi_type_index);
    return mpi_type_index;
  }

  bool pre_processing() override {
    internal_order_test();
    // Data TaskData  cite to type elements of vector input_
    if (world.rank() == 0) {
      delta_n = taskData->inputs_count[0] / world.size();
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
      local_input_ = std::vector<TypeElem>(input_.begin(), input_.begin() + delta_n);
    } else {
      local_input_ = std::vector<TypeElem>(delta_n);
    }
    MPI_Scatter(input_.data(), delta_n, get_mpi_type(), local_input_.data(), delta_n, get_mpi_type(), 0, world);
    result = {};
    left_index = {};
    right_index = 2;
    left_elem = {};
    right_elem = {};
    curr_index = {};
    return true;
  }
  bool validation() override {
    internal_order_test();
    // Check count elements of output
    if (world.rank() == 0) {
      return taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2;
    }
    return true;
  }
  bool run() override {  // everyone process will execute this code
    internal_order_test();
    // output local_input_ vector
    double local_result = 0;
    if (ops == "MAX_DIFFERENCE") {
      // here your algorithm task (.h files for task or all in run)
      local_result = max_difference();
    }
    if (ops == "MAX_DIFFERENCE") {
      // initialize buffers for MPI_Gather operation
      if (world.rank() == 0) {  // ONLY PROCESS 0
        global_result = std::vector<double>(world.size());
        global_indices = std::vector<TypeIndex>(world.size());
        ranks = std::vector<int>(world.size());
      }
      // everyone process send 1 element and get all local_results from everyone process 1 element
      double sendbuf[1];
      sendbuf[0] = local_result;
      MPI_Gather(sendbuf, 1, MPI_DOUBLE, global_result.data(), 1, MPI_DOUBLE, 0, world);
      TypeIndex sendbuf2[1];
      sendbuf2[0] = curr_index;
      MPI_Gather(sendbuf2, 1, get_mpi_type2(), global_indices.data(), 1, get_mpi_type2(), 0, world);
      int sendbuf3[1];
      sendbuf3[0] = world.rank();
      MPI_Gather(sendbuf3, 1, MPI_INT, ranks.data(), 1, MPI_INT, 0, world);
      // output global results
      if (world.rank() == 0) {
        calculate_global_delta();    // 1
        calculate_global_indices();  // 2
      }
    }
    // finalisation
    return true;
  }
  bool post_processing() override {  // proc 0 will write results
    internal_order_test();
    if (world.rank() == 0) {
      reinterpret_cast<TypeElem*>(taskData->outputs[0])[0] = left_elem;
      reinterpret_cast<TypeElem*>(taskData->outputs[0])[1] = right_elem;
      reinterpret_cast<TypeIndex*>(taskData->outputs[1])[0] = left_index;
      reinterpret_cast<TypeIndex*>(taskData->outputs[1])[1] = right_index;
    }
    return true;
  }

  ~TestMPITaskParallel() {
    // Освобождение MPI типов
    MPI_Type_free(&mpi_type_elem);
    MPI_Type_free(&mpi_type_index);
  }

 private:
  std::vector<TypeElem> input_;           // global vector
  std::vector<TypeElem> local_input_;     // local vector
  std::vector<double> global_result;      // global result
  std::vector<TypeIndex> global_indices;  // indices
  std::vector<int> ranks;                 // ranks for allocating max_delta everyone process
  unsigned int delta_n;
  double result;
  TypeIndex curr_index;
  TypeIndex left_index;
  TypeIndex right_index;
  TypeElem left_elem;
  TypeElem right_elem;
  std::string ops;
  boost::mpi::communicator world;
  MPI_Datatype mpi_type_elem;
  MPI_Datatype mpi_type_index;

 private:
  void print_local_data();
  double max_difference();
  void print_global_results();
  void calculate_global_delta();
  void calculate_global_indices();
};

template <typename TypeElem, typename TypeIndex>
void TestMPITaskParallel<TypeElem, TypeIndex>::print_local_data() {
  if (world.rank() == 0) {
    std::cout << "I'm proc 0" << "and my local_input data is ";
    for (unsigned int i = 0; i < delta_n; i++) {
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

template <typename TypeElem, typename TypeIndex>
double TestMPITaskParallel<TypeElem, TypeIndex>::max_difference() {
  // start delta between elements vector
  double max_delta = 0;
  double delta = 0;
  double local_result = 0;

  // get iterator for current element and his neighbor element vector
  auto iter_curr = local_input_.begin();
  auto iter_next = iter_curr + 1;
  auto iter_end = local_input_.end() - 1;
  auto iter_begin = local_input_.begin();
  // algorithm search max delta with using address arithmetic pointers
  while (iter_curr != iter_end) {
    delta = abs((double)(*iter_next - *iter_curr));
    if (delta > max_delta) {
      if (iter_begin == iter_curr) {
        curr_index = 0;
        max_delta = delta;
      } else {
        curr_index = static_cast<TypeIndex>(std::distance(local_input_.begin(), iter_curr));
        max_delta = delta;
      }
    }
    iter_curr++;
    iter_next = iter_curr + 1;
    // initialize results
    local_result = max_delta;
  }
  return local_result;
}
template <typename TypeElem, typename TypeIndex>
void TestMPITaskParallel<TypeElem, TypeIndex>::print_global_results() {
  if (world.rank() == 0) {
    result = 0;
    std::cout << std::endl;
    std::cout << "I`m proc" << world.rank() << " and global result vector is ";
    for (int i = 0; i < global_result.size(); i++) {
      std::cout << global_result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "global indices vector is ";
    for (int i = 0; i < global_indices.size(); i++) {
      std::cout << global_indices[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "ranks vector is ";
    for (int i = 0; i < ranks.size(); i++) {
      std::cout << ranks[i] << " ";
    }
  }
}

template <typename TypeElem, typename TypeIndex>
void TestMPITaskParallel<TypeElem, TypeIndex>::calculate_global_delta() {
  result = *std::max_element(global_result.begin(), global_result.end());
}

template <typename TypeElem, typename TypeIndex>
void TestMPITaskParallel<TypeElem, TypeIndex>::calculate_global_indices() {
  auto iter = std::find(global_result.begin(), global_result.end(), result);
  int ind = std::distance(global_result.begin(), iter);
  int r = ranks[ind];

  left_index = r * delta_n + global_indices[ind];
  right_index = left_index + 1;

  left_elem = input_[left_index];
  right_elem = input_[right_index];
}
}  // namespace kholin_k_vector_neighbor_diff_elems_mpi