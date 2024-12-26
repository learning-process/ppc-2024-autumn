#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace durynichev_d_radix_batcher_seq {

template <typename RandomIt, typename Compare = std::less<typename std::iterator_traits<RandomIt>::value_type>>
void radixSortDouble(RandomIt begin, RandomIt end, Compare comp = Compare{}) {
  using ValueType = typename std::iterator_traits<RandomIt>::value_type;
  static_assert(std::is_floating_point_v<ValueType>, "radixSortDouble requires floating-point values.");

  if (begin == end) {
    return;
  }

  constexpr size_t elemSize = sizeof(ValueType);
  constexpr size_t bitCount = elemSize * 8;
  constexpr size_t bucketCount = 256;

  auto toUint64Key = [](ValueType val) {
    static_assert(sizeof(ValueType) == sizeof(uint64_t), "Expected 64-bit floating-point type.");
    uint64_t u;
    memcpy(&u, &val, sizeof(uint64_t));

    if (u & (1ULL << (bitCount - 1))) {
      u = ~u;
    } else {
      u |= (1ULL << (bitCount - 1));
    }
    return u;
  };

  auto getByte = [](uint64_t key, size_t byteIndex) -> uint8_t {
    return static_cast<uint8_t>((key >> (byteIndex * 8)) & 0xFF);
  };

  auto length = static_cast<size_t>(std::distance(begin, end));
  std::vector<ValueType> temp(length);
  std::vector<size_t> count(bucketCount, 0);

  for (size_t bytePos = 0; bytePos < elemSize; ++bytePos) {
    std::fill(count.begin(), count.end(), 0);

    for (auto it = begin; it != end; ++it) {
      uint64_t key = toUint64Key(*it);
      ++count[getByte(key, bytePos)];
    }

    if (comp(ValueType(1), ValueType(0))) {
      for (size_t i = bucketCount - 1; i > 0; --i) {
        count[i - 1] += count[i];
      }
    } else {
      for (size_t i = 1; i < bucketCount; ++i) {
        count[i] += count[i - 1];
      }
    }

    for (auto it = end; it != begin;) {
      --it;
      uint64_t key = toUint64Key(*it);
      size_t pos = --count[getByte(key, bytePos)];
      temp[pos] = *it;
    }

    std::copy(temp.begin(), temp.end(), begin);
  }
}

class RadixBatcher : public ppc::core::Task {
 public:
  explicit RadixBatcher(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> arr;
};

}  // namespace durynichev_d_radix_batcher_seq
