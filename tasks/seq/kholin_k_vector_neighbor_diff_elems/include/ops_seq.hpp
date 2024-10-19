#pragma once

#include <string>
#include <vector>


#include "core/task/include/task.hpp"

using namespace std::chrono_literals;


namespace kholin_k_vector_neighbor_diff_elems_seq {
template <class TypeElem, class TypeIndex>
class MostDiffNeighborElements : public ppc::core::Task {
 public:
  explicit MostDiffNeighborElements(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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
    //
    // start delta between elements vector
    double max_delta = 0;
    double delta = 0;
    size_t curr_index = 0;
    // get iterator for current element and his neighbor element vector
    auto iter_curr = input_.begin();
    auto iter_next = iter_curr+1;
    auto iter_end = input_.end();
    auto iter_begin = input_.begin();
    // algorithm search max delta with using address arithmetic pointers
    while (iter_curr != (iter_end-1)) {
      delta = fabs(*iter_next - *iter_curr);
      if (delta > max_delta) {
        if (iter_begin == iter_curr) {
          curr_index = 0;
          max_delta = delta;
        } 
        else {
          curr_index = std::distance(input_.begin(), iter_curr);
          max_delta = delta;
        }
      }
      iter_curr++;
      iter_next = iter_curr + 1;
    }
    //initialize results
    result = max_delta;
    //std::cout << result; //max delta here
    right_index = curr_index + 1;
    left_index = curr_index;
    left_elem = input_[left_index];

    right_elem = input_[right_index];
    //std::cout << "left el " << left_elem << "left_ind " << left_index << std::endl;
    //std::cout << "right el" << right_elem << "right_ind" << right_index << std::endl;
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
};
}
// namespace kholin_k_vector_neighbour_diff_elems_seq

   
 

    
    

    

