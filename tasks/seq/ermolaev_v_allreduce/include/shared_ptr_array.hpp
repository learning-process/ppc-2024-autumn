#pragma once

#include <memory>

namespace ermolaev_v_allreduce_seq {

template <typename _T, typename _S = uint32_t>
class shared_ptr_array {
 public:
  shared_ptr_array() {}
  shared_ptr_array(_S size) {
    ptr_ = std::shared_ptr<_T>(new _T[size], [](_T* ptr) { delete[] ptr; });
  }

  _T& operator[](_S index) { return ptr_.get()[index]; }
  _T* get() { return ptr_.get(); }

 private:
  std::shared_ptr<_T> ptr_;
};

}  // namespace ermolaev_v_allreduce_seq