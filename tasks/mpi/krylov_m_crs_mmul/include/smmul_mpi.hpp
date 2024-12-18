#pragma once

#include <boost/serialization/vector.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "./matrix.hpp"
#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gather.hpp"
#include "boost/mpi/communicator.hpp"
#include "boost/serialization/array_wrapper.hpp"
#include "core/task/include/task.hpp"

namespace krylov_m_crs_mmul {

template <typename T>
class TaskCommon : public ppc::core::Task {
 public:
  explicit TaskCommon(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override {
    internal_order_test();

    return taskData->inputs.size() == 2 && taskData->inputs_count.size() == 4 && taskData->outputs.size() == 1 &&
           // (lhs.cols == rhs.rows)
           (taskData->inputs_count[1] == taskData->inputs_count[2]) &&
           // lhs.rows > 0 && lhs.cols > 0 && rhs.rows > 0 [&& rhs.cols > 0] - true by definition
           (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0);
  }

  bool pre_processing() override {
    internal_order_test();

    input = {*reinterpret_cast<CRSMatrix<T>*>(taskData->inputs[0]),
             reinterpret_cast<CRSMatrix<T>*>(taskData->inputs[1])->transpose()};

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    *reinterpret_cast<CRSMatrix<T>*>(taskData->outputs[0]) = res;

    return true;
  }

 protected:
  std::pair<CRSMatrix<T>, CRSMatrix<T>> input;
  CRSMatrix<T> res;
};

template <typename T, typename index_type = size_t>
struct CRSMatrixChunk {
  using index_type_mut = std::remove_const_t<index_type>;

  std::span<index_type> row_pointers;
  std::span<index_type> col_indices;
  std::span<T> data;
  index_type_mut cols;
  //
  struct {
    index_type_mut row;
    index_type_mut idx;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar & row;
      ar & idx;
    }
  } offsets;

  std::optional<CRSMatrix<std::remove_const_t<T>>> storage{std::nullopt};

  static CRSMatrixChunk from(std::span<index_type> row_pointers_, std::span<index_type> col_indices_,
                             std::span<T> data_, size_t cols_, const std::pair<index_type, index_type>& roff,
                             const std::pair<index_type, index_type>& off) {
    const auto& [rbegin, rend] = roff;
    const auto& [begin, end] = off;
    return {.row_pointers = {row_pointers_.begin() + rbegin, rend - rbegin + 1},
            .col_indices = {col_indices_.begin() + begin, end - begin},
            .data = {data_.begin() + begin, end - begin},
            .cols = cols_,
            .offsets = {.row = rbegin, .idx = begin}};
  }

  friend std::ostream& operator<<(std::ostream& os, const CRSMatrixChunk& m) {
    os << "CRSc(" << m.row_pointers.size() << "," << m.cols << "): ";
    os << "R/[";
    utils::iprint(os, m.row_pointers.begin(), m.row_pointers.end());
    os << "], C/[";
    utils::iprint(os, m.col_indices.begin(), m.col_indices.end());
    os << "], D/[";
    utils::iprint(os, m.data.begin(), m.data.end());
    os << "]";
    return os;
  }

 public:
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    size_t rows = row_pointers.size();
    size_t nz = data.size();
    //
    ar & rows;
    ar & nz;

    if constexpr (Archive::is_loading::value) {
      storage.emplace();
      //
      storage->row_pointers.resize(rows);
      storage->col_indices.resize(nz);
      storage->data.resize(nz);
      //
      row_pointers = storage->row_pointers;
      col_indices = storage->col_indices;
      data = storage->data;
    }

    ar & cols;
    ar & offsets;
    ar& boost::serialization::make_array(as_mut(row_pointers.data()), row_pointers.size());
    ar& boost::serialization::make_array(as_mut(col_indices.data()), col_indices.size());
    ar& boost::serialization::make_array(as_mut(data.data()), data.size());
  }

 private:
  // better to put it as a lambda to ::serialize, but compiler in CI is not fresh enough
  template <typename V>
  static constexpr V* as_mut(const V* v) {
    return const_cast<V*>(v);
  };
};

template <typename T>
class TaskParallel : public TaskCommon<T> {
 public:
  explicit TaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : TaskParallel::TaskCommon(std::move(taskData_)) {}

  bool validation() override {
    if (world.rank() != 0) {
      this->internal_order_test();
      return true;
    }
    return TaskParallel::TaskCommon::validation();
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

    boost::mpi::broadcast(world, this->input.second, 0);
    //
    CRSMatrixChunk<const T, const size_t> partial_lhs;
    const auto& rhs = this->input.second;

    if (world.rank() == 0) {
      const auto& lhs = this->input.first;

      const auto distrib = distribute(lhs.rows(), world.size());

      const auto initial_roff = std::make_pair(0, distrib[0]);
      auto roff = initial_roff;
      auto& [rbegin, rend] = roff;

      const auto world_size = world.size();
      for (int p = 1; p < world_size; p++) {
        rbegin = rend;
        rend += distrib[p];

        const auto off = std::make_pair(lhs.row_pointers[rbegin], lhs.row_pointers[rend]);
        partial_lhs = decltype(partial_lhs)::from(lhs.row_pointers, lhs.col_indices, lhs.data, lhs.cols_, roff, off);
        world.send(p, 0, partial_lhs);
      }
      partial_lhs = decltype(partial_lhs)::from(
          lhs.row_pointers, lhs.col_indices, lhs.data, lhs.cols_, initial_roff,
          std::make_pair(lhs.row_pointers[initial_roff.first], lhs.row_pointers[initial_roff.second]));
    } else {
      world.recv(0, 0, partial_lhs);
    }

    const auto [partial_rows, cols] =
        std::make_pair(partial_lhs.row_pointers.size() - 1, rhs.rows());  // rhs was transposed
    const auto& [roff, ioff] = partial_lhs.offsets;
    CRSMatrix<T> partial_res(partial_rows, cols);

    for (size_t row = roff; row < roff + partial_rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        auto [il, ir] = std::make_pair(partial_lhs.row_pointers[row - roff], rhs.row_pointers[col]);
        T sum{};
        while (il < partial_lhs.row_pointers[row + 1 - roff] && ir < rhs.row_pointers[col + 1]) {
          if (partial_lhs.col_indices[il - ioff] < rhs.col_indices[ir]) {
            il++;
          } else if (partial_lhs.col_indices[il - ioff] > rhs.col_indices[ir]) {
            ir++;
          } else {  // ==
            sum += partial_lhs.data[il - ioff] * rhs.data[ir];
            il++;
            ir++;
          }
        }
        if (sum != 0) {
          partial_res.data.push_back(sum);
          partial_res.col_indices.push_back(col);
        }
      }
      partial_res.row_pointers[row + 1 - roff] = partial_res.data.size();
    }

    boost::mpi::gather(world, partial_res, res_partials, 0);

    return true;
  }

  bool post_processing() override {
    if (world.rank() != 0) {
      this->internal_order_test();
      return true;
    }

    const auto size = std::accumulate(res_partials.begin(), res_partials.end(), size_t(0),
                                      [](size_t acc, CRSMatrix<T> chunks) { return acc + chunks.data.size(); });

    // REQUIREMENT: CRSMatrix::row_pointers capacity should be adjusted by the constructor below
    this->res = decltype(this->res)(this->input.first.rows(), this->input.second.rows());  // rhs was transposed
    this->res.col_indices.resize(size);
    this->res.data.resize(size);

    auto it_rowp = this->res.row_pointers.begin() + 1;
    auto it_cols = this->res.col_indices.begin();
    auto it_data = this->res.data.begin();
    //
    size_t nz = 0;
    for (const auto& chunk : res_partials) {
      auto prev_it_rowp = it_rowp;

      it_rowp = std::copy(chunk.row_pointers.begin() + 1, chunk.row_pointers.end(), it_rowp);
      it_cols = std::copy(chunk.col_indices.begin(), chunk.col_indices.end(), it_cols);
      it_data = std::copy(chunk.data.begin(), chunk.data.end(), it_data);

      for (auto& rp : std::ranges::subrange{prev_it_rowp, it_rowp}) {
        rp += nz;
      }
      nz = *(it_rowp - 1);
    }

    return TaskParallel::TaskCommon::post_processing();
  }

 private:
  static std::vector<size_t> distribute(size_t amount, size_t subject) {
    const auto avg = amount / subject;
    const auto ext = amount % subject;

    std::vector<size_t> distr(subject, avg);
    std::for_each(distr.begin(), distr.begin() + ext, [](auto& e) { ++e; });

    return distr;
  }

  boost::mpi::communicator world;
  std::vector<CRSMatrix<T>> res_partials;
};

template <class T>
void fill_task_data(ppc::core::TaskData& data, const CRSMatrix<T>& lhs, const CRSMatrix<T>& rhs, CRSMatrix<T>& out) {
  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&lhs)));
  data.inputs_count.emplace_back(lhs.rows());
  data.inputs_count.emplace_back(lhs.cols());
  //
  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&rhs)));
  data.inputs_count.emplace_back(rhs.rows());
  data.inputs_count.emplace_back(rhs.cols());

  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
}

}  // namespace krylov_m_crs_mmul
