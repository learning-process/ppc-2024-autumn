// Copyright 2024 Nesterov Alexander
#include "seq/kovalev_k_num_of_orderly_violations/include/header.hpp"

template <class T>
bool kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<T>::count_num_of_orderly_violations_seq() {
  res=0;	
  for(size_t i=1; i<n; i++)
	  if(v[i-1] > v[i])
		  res++;	  
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<T>::pre_processing() {
  internal_order_test();  
  // Init vector<T> of size n
  v = std::vector<T>(n);  
  void* ptr_input = taskData->inputs[0];
  void* ptr_vec = v.data();  
  memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  // Init counter
  res = 0;
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<T>::validation() {
  internal_order_test();  
  // Check count elements of output
  return (taskData->inputs_count[0] == n && taskData->outputs_count[0] == 1);
}

template <class T>
bool kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<T>::run() {
  internal_order_test();
  return count_num_of_orderly_violations_seq();
}

template <class T>
bool kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<T>::post_processing() {
  internal_order_test();  
  reinterpret_cast<size_t*>(taskData->outputs[0])[0] = res;
  return true;
}

template class kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int>;
template class kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<double>;