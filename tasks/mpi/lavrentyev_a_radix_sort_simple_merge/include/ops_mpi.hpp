#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace lavrentyev_a_radix_sort_simple_merge_mpi {

struct Node {
  double value;
  int source_rank;

  bool operator>(const Node& other) const { return value > other.value; }
};

void mergeSortedVectorsInPlace(std::vector<double>& arr, const std::vector<int>& sizes, const std::vector<int>& displs);

template <typename RandomIt>
void radixSortDouble(RandomIt begin, RandomIt end) {
  if (begin == end) return;

  using ValueType = typename std::iterator_traits<RandomIt>::value_type;
  static_assert(std::is_floating_point_v<ValueType>, "This function is designed for floating-point types.");

  const size_t numBits = sizeof(ValueType) * 8;
  const size_t numBuckets = 256;

  union DoubleIntUnion {
    ValueType d;
    uint64_t u;
  };

  std::vector<ValueType> buffer(std::distance(begin, end));
  std::vector<size_t> count(numBuckets);

  for (size_t byte = 0; byte < sizeof(ValueType); ++byte) {
    std::fill(count.begin(), count.end(), 0);

    for (auto it = begin; it != end; ++it) {
      DoubleIntUnion di;
      di.d = *it;
      uint64_t value = di.u;
      if (value & (1ULL << (numBits - 1))) {
        value = ~value;
      } else {
        value |= (1ULL << (numBits - 1));
      }
      uint8_t byteValue = (value >> (byte * 8)) & 0xFF;
      ++count[byteValue];
    }

    for (size_t i = 1; i < numBuckets; ++i) {
      count[i] += count[i - 1];
    }

    for (auto it = end; it != begin;) {
      --it;
      DoubleIntUnion di;
      di.d = *it;
      uint64_t value = di.u;
      if (value & (1ULL << (numBits - 1))) {
        value = ~value;
      } else {
        value |= (1ULL << (numBits - 1));
      }
      uint8_t byteValue = (value >> (byte * 8)) & 0xFF;
      buffer[--count[byteValue]] = *it;
    }

    std::copy(buffer.begin(), buffer.end(), begin);
  }
}

class RadixSimpleMerge : public ppc::core::Task {
 public:
  explicit RadixSimpleMerge(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int wsize;
  int wrank;
  int vsize;
  std::vector<double> arr;
  std::vector<int> sizes;
  std::vector<int> displs;
  std::vector<double> local_arr;
  boost::mpi::communicator world;
};

}  // namespace lavrentyev_a_radix_sort_simple_merge_mpi
