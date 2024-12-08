#include "mpi/agafeev_s_max_of_vector_elements/include/ops_mpi.hpp"

#include <iostream>
#include <limits>

// #include "boost/mpi/operations.hpp"

namespace agafeev_s_max_of_vector_elements_mpi {

template <typename T>
bool MaxMatrixSeq<T>::pre_processing() {
  internal_order_test();

  // Init value
  auto* temp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);

  return true;
}

template <typename T>
bool MaxMatrixSeq<T>::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

template <typename T>
bool MaxMatrixSeq<T>::run() {
  internal_order_test();

  maxres_ = get_MaxValue(input_);

  return true;
}

template <typename T>
bool MaxMatrixSeq<T>::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = maxres_;

  return true;
}

// Parallel

template <typename T>
bool MaxMatrixMpi<T>::pre_processing() {
  internal_order_test();
  maxres_ = std::numeric_limits<T>::min();

  return true;
}

template <typename T>
bool MaxMatrixMpi<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) return (taskData->outputs_count[0] == 1 && !(taskData->inputs.empty()));

  return true;
}

template <typename T>
bool MaxMatrixMpi<T>::run() {
  internal_order_test();

  unsigned int world_rank = world.rank();
  unsigned int world_size = world.size();
  unsigned int data_size = 0;

  if (world_rank == 0) {
    data_size = taskData->inputs_count[0];
    auto* temp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);
    /*for (unsigned long i = 0; i < input_.size(); ++i) {
      std::cout << "obshaya: " << input_[i] << std::endl;
    }*/
  }

  boost::mpi::broadcast(world, data_size, 0);

  unsigned int task_size = data_size / world_size;
  unsigned int over_size = data_size % world_size;
  lv_size = task_size;

  std::vector<int> sizes(world_size, task_size);
  std::vector<int> displs(world_size, 0);

  if (world_rank < (data_size % world_size)) {
    lv_size++;
  }

  if (world_rank == 0) {
    for (unsigned int i = 0; i < over_size; ++i) sizes[i]++;
    for (unsigned int i = 1; i < world_size; ++i) displs[i] = displs[i - 1] + sizes[i - 1];
  }

  local_vector.resize(lv_size);

  // boost::mpi::scatterv(world, input_, sizes, displs, local_vector.data(), lv_size, 0);
  if (world_rank == 0) {
    boost::mpi::scatterv(world, input_, sizes, displs, local_vector.data(), lv_size, 0);
  } else {
    boost::mpi::scatterv(world, local_vector.data(), lv_size, 0);
  }

  // std::cout << "my_rank" << world_rank << std::endl;
  // for (int i = 0; i < lv_size; ++i) {
  //   std::cout << local_vector[i] << "   ";
  // }
  // std::cout << std::endl;

  T res = agafeev_s_max_of_vector_elements_mpi::get_MaxValue<T>(local_vector);
  // std::cout<<"maxes: "<<res<<std::endl;
  boost::mpi::reduce(world, res, maxres_, boost::mpi::maximum<T>(), 0);
  // if (world_rank ==0 ){
  //   std::cout<<"MSXRES_"<<maxres_<<std::endl;
  // }
  // std::cout<<world_rank;
  // world.barrier();

  return true;
}

template <typename T>
bool MaxMatrixMpi<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) reinterpret_cast<T*>(taskData->outputs[0])[0] = maxres_;

  return true;
}

template class MaxMatrixMpi<int>;
}  // namespace agafeev_s_max_of_vector_elements_mpi
