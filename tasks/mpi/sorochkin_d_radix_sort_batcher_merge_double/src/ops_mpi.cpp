#include "../include/ops_mpi.hpp"

#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

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

template <class Compare>
void radix_sort(std::span<double> input, Compare cmp) {
  constexpr size_t base = 1 << CHAR_BIT;

  std::vector<double> aux_(input.size());
  std::span<double> aux{aux_};
  std::vector<size_t> count(base, 0);

  for (size_t ib = 0; ib < bytes<double>(); ++ib) {
    std::fill(count.begin(), count.end(), 0);
    std::for_each(input.begin(), input.end(),
                  [&](double el) { ++count[bitutil::byte_at(bitutil::as_uint64(el), ib)]; });

    if constexpr (cmp(0, 1)) {
      for (size_t i = 1; i < base; ++i) {
        count[i] += count[i - 1];
      }
    } else {
      for (size_t i = base - 1; i > 0; --i) {
        count[i - 1] += count[i];
      }
    }

    std::for_each(input.rbegin(), input.rend(),
                  [&](double el) { aux[--count[bitutil::byte_at(bitutil::as_uint64(el), ib)]] = el; });
    std::swap(input, aux);
  }
}

bool sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    const auto* src = reinterpret_cast<double*>(taskData->inputs[0]);
    input_.assign(src, src + taskData->inputs_count[0]);

    // the algorithms requires the array to be bitonic-bipartitioned
    const auto mid = input_.size() / 2;
    radix_sort(std::span{input_}.subspan(0, mid), std::less<>());
    radix_sort(std::span{input_}.subspan(mid), std::greater<>());
  }

  return true;
}

bool sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() != 0) {
    return true;
  }

  const auto n = taskData->inputs_count[0];
  return (n & (n - 1)) == 0 && n == taskData->inputs_count[0];
}

bool sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  size_t amount = input_.size();
  boost::mpi::broadcast(world, amount, 0);

  const auto mp2 = std::pow(2, std::floor(std::sqrt(world.size())));
  if (world.rank() >= mp2 || static_cast<size_t>(world.rank()) >= amount) {
    world.split(1);
    return true;
  }
  auto group = world.split(0);

  const int remainder = amount % group.size();
  std::vector<int> sizes(group.size(), amount / group.size());
  for (int i = 0; i < remainder; i++) {
    ++sizes[i];
  }

  const size_t local_size = sizes[group.rank()];
  std::vector<double> local_input(local_size);
  std::vector<double> aux(local_size);

  boost::mpi::scatterv(group, input_, sizes, local_input.data(), 0);

  const auto ws = group.size();
  std::vector<bool> active_workers(ws);  // vector<bool> is implemented as a bitset actually

  for (int i = ws / 2; i > 0; i /= 2) {
    std::fill(active_workers.begin(), active_workers.end(), false);

    for (int rank = 0; rank < ws; rank++) {
      if (rank - i < 0 || (rank + i < ws && !active_workers[rank - i])) {
        active_workers[rank] = true;
      }
    }

    if (active_workers[group.rank()]) {
      group.send(group.rank() + i, 0, local_input);
      group.recv(group.rank() + i, 1, local_input);
    } else {
      group.recv(group.rank() - i, 0, aux);

      for (size_t j = 0; j < local_input.size(); ++j) {
        if (aux[j] > local_input[j]) {
          std::swap(aux[j], local_input[j]);
        }
      }

      group.send(group.rank() - i, 1, aux);
    }
  }

  radix_sort(local_input, std::less<>());

  if (group.rank() == 0) {
    res_.resize(input_.size());
  }
  boost::mpi::gatherv(group, local_input, res_.data(), sizes, 0);

  return true;
}

bool sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }
  return true;
}
