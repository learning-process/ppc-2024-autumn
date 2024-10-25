#include "mpi/beskhmelnova_k_most_different_neighbor_elements/include/mpi.hpp"

template <typename DataType>
std::vector<DataType> beskhmelnova_k_most_different_neighbor_elements_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<DataType> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

template <typename DataType>
int beskhmelnova_k_most_different_neighbor_elements_mpi::position_of_first_neighbour_seq(std::vector<DataType> vector) {
  int n = vector.size();
  if (n == 0 || n == 1) return -1;
  DataType max_dif = abs(vector[0] - vector[1]);
  DataType dif;
  int index = 0;
  for (int i = 1; i < n - 1; i++) {
    dif = abs(vector[i] - vector[i + 1]);
    if (dif > max_dif) {
      max_dif = dif;
      index = i;
    }
  }
  return index;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskSequential<DataType>::pre_processing() {
  internal_order_test();
  // Init value for input
  int n = taskData->inputs_count[0];
  input_ = std::vector<DataType>(n);
  void* ptr_r = taskData->inputs[0];
  void* ptr_d = input_.data();
  memcpy(ptr_d, ptr_r, sizeof(DataType) * n);
  return true;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskSequential<DataType>::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 2;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskSequential<DataType>::run() {
  internal_order_test();
  int index = position_of_first_neighbour_seq(input_);
  if (index == -1) {
    res[0] = -1;
    res[1] = -1;
    return true;
  }
  res[0] = input_[index];
  res[1] = input_[index + 1];
  return true;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskSequential<DataType>::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res[0];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = res[1];
  return true;
}

//// Struct of 2 most different neighbour elements
// template <typename DataType>
// struct NeighborDifference {
//   DataType first;
//   DataType second;
//   DataType dif;
//
//   // Serialization of the structure for MPI transmission
//   template <class Archive>
//   void serialize(Archive& ar, const unsigned int version) {
//     ar & first;
//     ar & second;
//     ar & dif;
//   }
// };
//
//// Find the most different neighbors in a vector
// template <typename DataType>
// NeighborDifference<DataType> find_max_difference(const std::vector<DataType> vector) {
//   int n = vector.size();
//   if (n == 0 || n == 1) return NeighborDifference<DataType>{1, 1, -1};
//   NeighborDifference<DataType> max_dif = {vector[0], vector[1], std::abs(vector[1] - vector[0])};
//   for (unsigned long i = 1; i < vector.size() - 1; i++) {
//     DataType dif = std::abs(vector[i + 1] - vector[i]);
//     if (dif > max_dif.dif) {
//       max_dif = {vector[i], vector[i + 1], dif};
//     }
//   }
//   return max_dif;
// }
//
//// Functor for a custom operation in reduce
// template <typename DataType>
// struct reduce_max_difference {
//   NeighborDifference<DataType> operator()(const NeighborDifference<DataType>& a,
//                                           const NeighborDifference<DataType>& b) const {
//     return (a.dif > b.dif) ? a : b;
//   }
// };

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<DataType>::pre_processing() {
  internal_order_test();
  int world_rank = world.rank();
  int world_size = world.size();
  int n;
  if (world_rank == 0) {
    n = taskData->inputs_count[0];
    input_ = std::vector<DataType>(n);
    void* ptr_r = taskData->inputs[0];
    void* ptr_d = input_.data();
    memcpy(ptr_d, ptr_r, sizeof(DataType) * n);
  }
  broadcast(world, n, 0);
  int send_size = n / world_size;
  int over_size = n % world_size;
  std::vector<int> send_counts(world_size, send_size);
  std::vector<int> offset(world_size, 0);
  if (world_rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      if (i < over_size) send_counts[i]++;
      if (send_counts[i] % 2 != 0) send_counts[i]++;
      if (i > 0) offset[i] = offset[i - 1] + send_counts[i - 1];
    }
  }
  int local_vector_size = send_counts[world_rank];
  local_input_.resize(local_vector_size);
  if (world.rank() == 0)
    boost::mpi::scatterv(world, input_, send_counts, offset, local_input_.data(), local_vector_size, 0);
  else
    boost::mpi::scatterv(world, local_input_.data(), local_vector_size, 0);
  local_input_size = local_vector_size;
  return true;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<DataType>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 2;
  }
  return true;
}

// template <typename DataType>
// bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<DataType>::run() {
//   internal_order_test();
//   if (local_input_size == 0 || taskData->inputs_count[0] == 0 || taskData->inputs_count[0] == 1) {
//     res[0] = -1;
//     res[1] = -1;
//     return true;
//   }
//   NeighborDifference<DataType> local_result = find_max_difference(local_input_);
//   NeighborDifference<DataType> global_result = {0, 0, 0};
//   reduce(world, local_result, global_result, reduce_max_difference<DataType>(), 0);
//   res[0] = global_result.first;
//   res[1] = global_result.second;
//   return true;
// }

// Struct of 2 most different neighbour elements
template <typename DataType>
struct NeighborDifference {
  DataType first;
  DataType second;
  DataType dif;
};

template <typename DataType>
NeighborDifference<DataType> find_max_difference(const std::vector<DataType>& vector) {
  int n = vector.size();
  if (n == 0 || n == 1) return {1, 1, -1};
  NeighborDifference<DataType> max_dif = {vector[0], vector[1], std::abs(vector[1] - vector[0])};
  for (int i = 1; i < n - 1; ++i) {
    DataType dif = std::abs(vector[i + 1] - vector[i]);
    if (dif > max_dif.dif) {
      max_dif = {vector[i], vector[i + 1], dif};
    }
  }
  return max_dif;
}

template <typename DataType>
void reduce_max_difference(const DataType* in_data, DataType* inout_data, int* len, MPI_Datatype* dptr) {
  if (in_data[2] > inout_data[2]) {
    inout_data[0] = in_data[0];
    inout_data[1] = in_data[1];
    inout_data[2] = in_data[2];
  }
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<DataType>::run() {
  internal_order_test();
  NeighborDifference<DataType> local_result = find_max_difference(local_input_);
  DataType local_data[3] = {local_result.first, local_result.second, local_result.dif};
  DataType global_data[3] = {0, 0, 0};
  MPI_Op custom_op;
  MPI_Op_create(reinterpret_cast<MPI_User_function*>(&reduce_max_difference<DataType>), 1, &custom_op);
  MPI_Reduce(local_data, global_data, 3, MPI_DOUBLE, custom_op, 0, MPI_COMM_WORLD);
  if (world.rank() == 0) {
    res[0] = global_data[0];
    res[1] = global_data[1];
  }
  MPI_Op_free(&custom_op);
  return true;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<DataType>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res[0];
    reinterpret_cast<int*>(taskData->outputs[0])[1] = res[1];
  }
  return true;
}
