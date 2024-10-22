#include "seq/beskhmelnova_k_most_different_neighbor_elements/include/seq.hpp"


template <typename DataType>
std::vector<DataType> beskhmelnova_k_most_different_neighbor_elements_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<DataType> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

template <typename DataType>
int beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<DataType>::position_of_first_neighbour_seq(std::vector<DataType> vector) {
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
bool beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<DataType>::pre_processing() {
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
bool beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<DataType>::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 2;
}

template <typename DataType>
bool beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<DataType>::run() {
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
bool beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<DataType>::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res[0];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = res[1];
  return true;
}