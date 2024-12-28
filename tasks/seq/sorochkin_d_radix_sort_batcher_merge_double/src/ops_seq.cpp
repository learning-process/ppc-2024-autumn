#include "../include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

template <typename T>
static constexpr size_t bytes() {
  return sizeof(T);
}
template <typename T>
static constexpr size_t bits() {
  return bytes<T>() * CHAR_BIT;
}
class bitutil {
 private:
  union du64 {
    double d;
    uint64_t u;
    static constexpr uint64_t MASK = 1ull << ((sizeof(uint64_t) * CHAR_BIT) - 1);
  };

 public:
  static constexpr uint64_t as_uint64(double x) {
    const du64 r{.d = x};
    return ((r.u & du64::MASK) != 0u) ? ~r.u : r.u | du64::MASK;
  }

  template <typename T>
    requires std::is_floating_point_v<T> or std::is_integral_v<T>
  static constexpr uint8_t byte_at(const T& val, uint8_t idx) {
    return (val >> (idx * 8)) & 0xFF;
  }
};

bool sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  const auto* src = reinterpret_cast<double*>(taskData->inputs[0]);
  input_.assign(src, src + taskData->inputs_count[0]);

  return true;
}

bool sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential::run() {
  internal_order_test();

  constexpr size_t base = 1 << CHAR_BIT;

  res_ = input_;
  decltype(res_) aux(input_.size());
  std::vector<size_t> count(base, 0);

  for (size_t ib = 0; ib < bytes<double>(); ++ib) {
    std::fill(count.begin(), count.end(), 0);
    std::for_each(res_.begin(), res_.end(), [&](double el) { ++count[bitutil::byte_at(bitutil::as_uint64(el), ib)]; });
    for (size_t i = 1; i < base; ++i) {
      count[i] += count[i - 1];
    }
    std::for_each(res_.rbegin(), res_.rend(),
                  [&](double el) { aux[--count[bitutil::byte_at(bitutil::as_uint64(el), ib)]] = el; });
    std::swap(res_, aux);
  }

  return true;
}

bool sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.begin(), res_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}
