#pragma once

#include <string>
#include <vector>


#include "core/task/include/task.hpp"

using namespace std::chrono_literals;


namespace kholin_k_vector_neighbor_diff_elems_seq {
template <class TypeElem>
class MostDiffNeighborElements : public ppc::core::Task {
 public:
  explicit MostDiffNeighborElements(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override {
    internal_order_test();
    // Data TaskData  cite to type elements of vector input_
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = *(reinterpret_cast<TypeElem*>(taskData->inputs[i]));
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
    //
    // start delta between elements vector
    double max_delta = 0;
    double delta = 0;
    size_t curr_index = 0;

    // get iterator for current element and his neighbor elements vector
    auto iter_prev = input_.begin();
    auto iter_curr = iter_prev + 1;
    auto iter_next = iter_curr + 1;
    auto iter_end = input_.end();

    // algorithm search max delta with using address arithmetic pointers
    while (iter_curr != iter_end - 1) {
      delta = abs(*iter_next - *iter_prev);
      if (delta > max_delta) {
        curr_index = std::distance(input_.begin(), iter_curr);
        max_delta = delta;
      }
      iter_curr++;
      if (iter_curr == iter_end) {
        break;
      }
      iter_prev = iter_curr - 1;
      iter_next = iter_curr + 1;
    }
    result = max_delta;
    right_index = curr_index + 1;
    left_index = curr_index - 1;

    return true;
  }
  // get results
  bool post_processing() override {
    internal_order_test();
    reinterpret_cast<TypeElem*>(taskData->outputs[0])[0] = left_elem;
    reinterpret_cast<TypeElem*>(taskData->outputs[0])[1] = right_elem;
    reinterpret_cast<TypeElem*>(taskData->outputs[1])[0] = left_index;
    reinterpret_cast<TypeElem*>(taskData->outputs[1])[1] = right_index;
    return true;
  }

 private:
  std::vector<TypeElem> input_; 
  double result;
  size_t left_index;
  size_t right_index;
  TypeElem left_elem;
  TypeElem right_elem;
};
}
// namespace kholin_k_vector_neighbour_diff_elems_seq

   
 

    
    

    

