#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <unordered_set>

#include "core/task/include/task.hpp"

namespace durynichev_d_radix_batcher_mpi {
void exchange_data(boost::mpi::communicator& comm, int shift, std::vector<double>& data, bool is_sending);
std::unordered_set<int> populate_send_workers(int comm_size, int shift);
void batcher(boost::mpi::communicator& comm, std::vector<double>& data);

template <typename RandomIt, typename Compare = std::less<typename std::iterator_traits<RandomIt>::value_type>>
void radixSortDouble(RandomIt begin, RandomIt end, Compare comp = Compare{}) {
  using ValueType = typename std::iterator_traits<RandomIt>::value_type;

  if (begin == end) {
    return;
  }

  constexpr size_t elemSize = sizeof(ValueType);
  constexpr size_t bitCount = elemSize * 8;
  constexpr size_t bucketCount = 256;

  union DoubleInt64 {
    ValueType d;
    uint64_t u;
  };

  auto toUint64Key = [](ValueType val) -> uint64_t {
    DoubleInt64 di{val};
    uint64_t u = di.u;

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
  std::vector<ValueType> buffer(length);
  RandomIt src = begin;
  RandomIt dst = buffer.begin();

  bool reverse = comp(ValueType(1), ValueType(0));

  for (size_t bytePos = 0; bytePos < elemSize; ++bytePos) {
    std::vector<size_t> count(bucketCount, 0);

    for (auto it = src; it != src + length; ++it) {
      uint64_t key = toUint64Key(*it);
      ++count[getByte(key, bytePos)];
    }

    if (reverse) {
      for (size_t i = bucketCount - 1; i > 0; --i) {
        count[i - 1] += count[i];
      }
    } else {
      for (size_t i = 1; i < bucketCount; ++i) {
        count[i] += count[i - 1];
      }
    }

    for (auto it = src + length; it != src;) {
      --it;
      uint64_t key = toUint64Key(*it);
      size_t pos = --count[getByte(key, bytePos)];
      dst[pos] = *it;
    }

    std::swap(src, dst);
  }

  if (src != begin) {
    std::copy(src, src + length, begin);
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
  int arr_size;
  int workers_count;
  std::vector<double> input_data;
  std::vector<double> output_data;
  boost::mpi::communicator world;
};

}  // namespace durynichev_d_radix_batcher_mpi
