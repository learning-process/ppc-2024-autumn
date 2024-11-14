#pragma once

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ostream>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace krylov_m_matmul_strip_ha_vb_mpi {

template <class T, std::integral size_type_ = size_t, bool view_ = false>
struct TMatrix {
  using size_type = size_type_;

  // template<template<typename, typename...>> for TMatrix is not suitable as span has value template argument
  template <class TT>
  using Container = std::conditional_t<view_, std::span<TT>, std::vector<TT>>;

  using View = TMatrix<T, size_type, true>;

  size_type rows;
  size_type cols;
  Container<T> storage{};

  operator View() { return {rows, cols, std::span{storage}}; }

  bool check_integrity() const noexcept { return storage.size() == static_cast<size_t>(rows * cols); }

  inline const T& at(size_type row, size_type col) const noexcept {
    const auto idx = static_cast<size_t>(row * cols + col);
    assert(idx < storage.size());
    return storage[idx];
  }
  inline T& at(size_type row, size_type col) noexcept { return const_cast<T&>(std::as_const(*this).at(row, col)); }

  bool operator==(const TMatrix& other) const noexcept {
    return rows == other.rows && cols == other.cols && storage == other.storage;
  }

  void read(const T* src) { storage.assign(src, src + rows * cols); }

  friend std::ostream& operator<<(std::ostream& os, const TMatrix& m) {
    os << "M(" << m.rows << "," << m.cols << "): [";
    for (const auto& e : m.storage | std::views::take(m.storage.size() - 1)) {
      os << e << ' ';
    }
    os << m.storage.back() << ']';
    return os;
  }

  static TMatrix create(size_type rows, size_type cols) {
    static_assert(!view_);
    return {rows, cols, Container<T>(rows * cols)};
  }
};

template <class T, std::integral size_type = int>
using TMatrixView = TMatrix<T, size_type, true>;

template <class T>
class TaskCommon : public ppc::core::Task {
 public:
  using Matrix = TMatrix<T>;

  explicit TaskCommon(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override {
    internal_order_test();

    return taskData->inputs_count.size() == 4 && taskData->outputs.size() == 3 && taskData->outputs_count.size() == 3 &&
           // (lhs.cols == rhs.rows)
           (taskData->inputs_count[1] == taskData->inputs_count[2]) &&
           // lhs.rows > 0 && lhs.cols > 0 && rhs.rows > 0 [&& rhs.cols > 0] - true by definition
           (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0) &&
           // out.storage.size() == lhs.rows * rhs.cols
           (taskData->outputs_count[0] == taskData->inputs_count[0] * taskData->inputs_count[3]) &&
           // out.rows, out.cols
           (taskData->outputs_count[1] == 1 && taskData->outputs_count[2] == 1);
  }

  bool pre_processing() override {
    internal_order_test();

    input = std::make_pair<Matrix, Matrix>({taskData->inputs_count[0], taskData->inputs_count[1]},
                                           {taskData->inputs_count[2], taskData->inputs_count[3]});
    input.first.read(reinterpret_cast<T*>(taskData->inputs[0]));
    input.second.read(reinterpret_cast<T*>(taskData->inputs[1]));

    res.rows = input.first.rows;
    res.cols = input.second.cols;
    res.storage.resize(res.rows * res.cols);
    assert(res.check_integrity());

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    assert(res.check_integrity());
    *reinterpret_cast<size_t*>(taskData->outputs[1]) = res.rows;
    *reinterpret_cast<size_t*>(taskData->outputs[2]) = res.cols;
    std::copy(res.storage.begin(), res.storage.end(), reinterpret_cast<T*>(taskData->outputs[0]));

    return true;
  }

 protected:
  std::pair<Matrix, Matrix> input{};
  Matrix res;
};

template <class T>
class TaskSequential : public TaskCommon<T> {
  using dimen_t = TaskSequential::TaskCommon::Matrix::size_type;
  static_assert(std::is_integral_v<dimen_t>);

 public:
  explicit TaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : TaskSequential::TaskCommon(std::move(taskData_)) {}

  bool run() override {
    this->internal_order_test();

    const auto& [lhs, rhs] = this->input;

    for (dimen_t i = 0; i < lhs.rows; i++) {
      for (dimen_t j = 0; j < rhs.cols; j++) {
        this->res.at(i, j) = 0;
        for (dimen_t k = 0; k < rhs.rows; k++) {
          this->res.at(i, j) += lhs.at(i, k) * rhs.at(k, j);
        }
      }
    }

    return true;
  }
};

template <class T>
class TaskParallel : public TaskCommon<T> {
  using dimen_t = decltype(std::declval<boost::mpi::communicator>().rank());
  static_assert(std::is_integral_v<dimen_t>);

  using Matrix = TMatrix<T, dimen_t>;
  using MatrixView = Matrix::View;

 public:
  explicit TaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : TaskParallel::TaskCommon(std::move(taskData_)) {}

  bool validation() override {
    if (world.rank() != 0) {
      this->internal_order_test();
      return true;
    }

    constexpr auto maxdim = std::numeric_limits<dimen_t>::max();
    return TaskParallel::TaskCommon::validation() &&
           // lhs.rows > 0 && lhs.cols < maxdim && rhs.rows < maxdim [&& rhs.cols < maxdim] - true by definition
           this->taskData->inputs_count[0] < maxdim && this->taskData->inputs_count[1] < maxdim &&
           this->taskData->inputs_count[2] < maxdim;
  }

  bool pre_processing() override {
    if (world.rank() != 0) {
      this->internal_order_test();
      return true;
    }
    return TaskParallel::TaskCommon::pre_processing();
  }

  bool run() override {
    this->internal_order_test();

    const dimen_t workers = world.size();

    // god bless llvm/clang tools
    // - https://github.com/llvm/llvm-project/pull/80216
    // - https://github.com/llvm/llvm-project/issues/91835

    const auto globals /*[n, m, nr]*/ = [&]() {
      dimen_t n_ = this->input.first.cols;
      boost::mpi::broadcast(world, n_, 0);
      //
      dimen_t m_ = this->input.second.cols;
      boost::mpi::broadcast(world, m_, 0);
      //
      dimen_t nr_ = this->input.first.rows;
      boost::mpi::broadcast(world, nr_, 0);
      //
      return std::make_tuple(n_, m_, nr_);
    }();
    const auto n = std::get<0>(globals);  // cheaper to copy
    const auto m = std::get<1>(globals);
    const auto nr = std::get<2>(globals);

    const dimen_t np = std::min({nr, m, workers});
    const dimen_t si = world.rank();  // strip_index

    if (si >= np) {
      world.split(1);
      return true;
    }

    const auto resolve_distribution = [np, si](dimen_t tgt) {
      const dimen_t stride = std::max(1, distribution_for(tgt, np, si));
      const dimen_t under_threshold = (tgt % np) == 0 ? np : (tgt % np);
      const dimen_t lg = std::max(1, distribution_for(tgt, np, 0)) + (under_threshold == 0 ? 1 : 0);
      return std::make_tuple(stride, under_threshold,
                             [under_threshold, lg](dimen_t rank) { return (rank < under_threshold) ? lg : (lg - 1); });
    };

    const auto h_distrib /*[h_strip_stride, h_strip_stride_underload_th, h_strip_stride_of]*/ =
        resolve_distribution(nr);
    const auto h_strip_stride = std::get<0>(h_distrib);
    // const auto h_strip_stride_underload_th = std::get<1>(h_distrib);
    const auto& h_strip_stride_of = std::get<2>(h_distrib);
    const dimen_t h_strip_size = n * h_strip_stride_of(si);
    //
    const auto v_distrib /*[v_strip_stride, v_strip_stride_underload_th, v_strip_stride_of]*/ = resolve_distribution(m);
    const auto v_strip_stride = std::get<0>(v_distrib);
    const auto v_strip_stride_underload_th = std::get<1>(v_distrib);
    const auto& v_strip_stride_of = std::get<2>(v_distrib);
    const dimen_t v_strip_size = n * v_strip_stride_of(si);
    const auto v_strip_strides = distribute(m, np);

    std::vector<T> buf(h_strip_size + v_strip_size);
    //
    const MatrixView h{h_strip_stride, n, std::span{buf}.subspan(0, h_strip_size)};
    const MatrixView v{n, v_strip_stride, std::span{buf}.subspan(h_strip_size, v_strip_size)};

    if (world.rank() == 0) {
      // const auto& [lhs, rhs] = this->input;
      const auto& lhs = std::get<0>(this->input);
      const auto& rhs = std::get<1>(this->input);
      //

      const auto stripe_diff = [&](dimen_t rank) {
        return std::make_pair(n * h_strip_stride_of(rank), v_strip_strides[rank]);
      };
      const auto collect_stripes = [&](MatrixView mut_v, std::pair<dimen_t, dimen_t>& offsets,
                                       const std::pair<dimen_t, dimen_t>& diffs) {
        auto& [a_off, b_off] = offsets;
        const auto& [da, db] = diffs;

        std::copy(lhs.storage.begin() + a_off, lhs.storage.begin() + a_off + da, h.storage.begin());
        for (dimen_t i = 0; i < n; ++i) {
          for (dimen_t j = 0; j < db; ++j) {
            mut_v.at(i, j) = rhs.at(i, b_off + j);
          }
        }

        a_off += da;
        b_off += db;
      };

      auto offsets = stripe_diff(0);
      for (dimen_t p = 1; p < np; ++p) {
        const auto diffs = stripe_diff(p);
        const auto [hsst, vsst] = std::make_pair(h_strip_stride_of(p), v_strip_stride_of(p));
        collect_stripes({n, vsst, std::span{buf}.subspan(n * hsst, n * vsst)}, offsets, diffs);
        world.send(p, 0, buf.data(), diffs.first + n * diffs.second);
      }
      //
      offsets = {0, 0};
      collect_stripes(v, offsets, stripe_diff(0));
    } else {
      world.recv(0, 0, buf.data(), buf.size());
    }

    auto group = world.split(0);

    auto external_v = Matrix::create(n, v_strip_strides[0]);
    auto res_strip = Matrix::create(h_strip_stride, m);

    const auto calc_horizontal_offset_up_to = [&]() {
      const dimen_t extra_s = v_strip_strides[0];
      const dimen_t normal_s = v_strip_strides[np - 1];
      return [&np, extra_s, normal_s, &th = v_strip_stride_underload_th](dimen_t vid) {
        return extra_s * std::clamp(vid, 0, th) + normal_s * std::clamp(vid - th, 0, np - th);
      };
    }();

    const auto mul_hv = [&](dimen_t vid, MatrixView v_) {
      v_.cols = v_strip_strides[vid];

      const dimen_t h_off = calc_horizontal_offset_up_to(vid);

      for (dimen_t i = 0; i < h.rows; ++i) {
        for (dimen_t j = 0; j < v_.cols; ++j) {
          res_strip.at(i, j + h_off) = 0;
          for (dimen_t k = 0; k < v_.rows; ++k) {
            res_strip.at(i, j + h_off) += h.at(i, k) * v_.at(k, j);
          }
        }
      }
    };
    const auto recv_and_mul = [&](dimen_t begin, dimen_t end) {
      for (dimen_t i = begin; i < end; ++i) {
        boost::mpi::broadcast(group, external_v.storage.data(), n * v_strip_strides[i], i);
        mul_hv(i, external_v);
      }
    };

    mul_hv(si, v);
    recv_and_mul(0, si);
    boost::mpi::broadcast(group, v.storage.data(), v.storage.size(), si);
    recv_and_mul(si + 1, np);

    if (group.rank() == 0) {
      assert(this->res.check_integrity());

      auto sizes = distribute(nr, np);
      std::for_each(sizes.begin(), sizes.end(), [m](auto& e) { e *= m; });

      boost::mpi::gatherv(group, res_strip.storage.data(), res_strip.storage.size(), this->res.storage.data(), sizes,
                          0);
    } else {
      boost::mpi::gatherv(group, res_strip.storage.data(), res_strip.storage.size(), 0);
    }

    return true;
  }

  bool post_processing() override {
    if (world.rank() != 0) {
      this->internal_order_test();
      return true;
    }
    return TaskParallel::TaskCommon::post_processing();
  }

 private:
  static std::vector<dimen_t> distribute(dimen_t amount, dimen_t subject) {
    const auto avg = amount / subject;
    const auto ext = amount % subject;

    std::vector<dimen_t> distr(subject, avg);
    std::for_each(distr.begin(), distr.begin() + ext, [](auto& e) { ++e; });

    return distr;
  }
  static dimen_t distribution_for(dimen_t amount, dimen_t subject, dimen_t target) {
    const auto avg = amount / subject;
    const auto ext = amount % subject;

    return avg + ((target < ext) ? 1 : 0);
  }

  boost::mpi::communicator world;
};

template <class T>
void fill_task_data(ppc::core::TaskData& data, const TMatrix<T>& lhs, const TMatrix<T>& rhs, TMatrix<T>& out) {
  out.storage.resize(lhs.rows * rhs.cols);

  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(lhs.storage.data())));
  data.inputs_count.emplace_back(lhs.rows);
  data.inputs_count.emplace_back(lhs.cols);
  //
  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rhs.storage.data())));
  data.inputs_count.emplace_back(rhs.rows);
  data.inputs_count.emplace_back(rhs.cols);

  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(out.storage.data()));
  data.outputs_count.emplace_back(out.storage.size());
  //
  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(&out.rows));
  data.outputs_count.emplace_back(1);
  //
  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(&out.cols));
  data.outputs_count.emplace_back(1);
}

}  // namespace krylov_m_matmul_strip_ha_vb_mpi
