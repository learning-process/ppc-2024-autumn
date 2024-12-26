#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>

#include "core/task/include/task.hpp"

namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi {

template <typename DataType>
class MatrixMultiplicationParallel : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), taskData(taskData_) {}

  static std::vector<size_t> distribute(size_t amount, size_t subject) {
    const auto avg = amount / subject;
    const auto ext = amount % subject;

    std::vector<size_t> distr(subject, avg);
    std::for_each(distr.begin(), distr.begin() + ext, [](auto& e) { ++e; });

    return distr;
  }

  bool pre_processing() override {
    internal_order_test();

    if (world.rank() != 0) {
      return true;
    }

    m = taskData->inputs_count[0];
    k = taskData->inputs_count[1];
    n = taskData->inputs_count[2];

    auto tmp_ptr_A = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto tmp_ptr_B = reinterpret_cast<DataType*>(taskData->inputs[1]);
    A.assign(tmp_ptr_A, tmp_ptr_A + m * k);
    B.assign(tmp_ptr_B, tmp_ptr_B + k * n);

    C = std::vector<DataType>(m * n, 0);

    return true;
  }

  bool validation() override {
    internal_order_test();

    return (
        A.size() == m * k && B.size() == k * n &&
        (world.rank() != 0 || (taskData->inputs.size() == 2 && taskData->inputs_count.size() == 3 && m * k == n * k)));
  }

  bool run() override {
    internal_order_test();

    auto rank = static_cast<size_t>(world.rank());
    auto size = static_cast<size_t>(world.size());

    std::vector<size_t> distributions;

    if (rank == 0) {
      distributions = distribute(m, size);
    }

    boost::mpi::broadcast(world, distributions, 0);
    boost::mpi::broadcast(world, k, 0);
    boost::mpi::broadcast(world, n, 0);

    std::vector<DataType> local_A(distributions[rank] * k);

    if (rank == 0) {
      std::vector<int> sendcounts(size);
      std::vector<int> displs(size);

      for (size_t i = 0; i < size; ++i) {
        size_t proc_start_row = std::accumulate(distributions.begin(), distributions.begin() + i, 0);
        size_t proc_row_count = distributions[i];
        sendcounts[i] = static_cast<int>(proc_row_count * k);
        displs[i] = static_cast<int>(proc_start_row * k);
      }
      boost::mpi::scatterv(world, A.data(), sendcounts, displs, local_A.data(), local_A.size(), 0);
    } else {
      boost::mpi::scatterv(world, static_cast<DataType*>(nullptr), {}, {}, local_A.data(), local_A.size(), 0);
    }
    boost::mpi::broadcast(world, B, 0);

    std::vector<DataType> local_C(distributions[rank] * n, 0);

    for (size_t i = 0; i < distributions[rank]; ++i) {
      for (size_t j = 0; j < n; ++j) {
        for (size_t p = 0; p < k; ++p) {
          local_C[i * n + j] += local_A[i * k + p] * B[p * n + j];
        }
      }
    }
    if (rank == 0) {
      std::vector<int> recvcounts(size);
      std::vector<int> displs(size);

      for (size_t i = 0; i < size; ++i) {
        size_t proc_start_row = std::accumulate(distributions.begin(), distributions.begin() + i, 0);
        size_t proc_end_row = proc_start_row + distributions[i];
        size_t proc_row_count = proc_end_row - proc_start_row;

        recvcounts[i] = static_cast<int>(proc_row_count * n);
        displs[i] = static_cast<int>(proc_start_row * n);
      }

      boost::mpi::gatherv(world, local_C.data(), local_C.size(), C.data(), recvcounts, displs, 0);
    } else {
      boost::mpi::gatherv(world, local_C.data(), local_C.size(), C.data(), {}, {}, 0);
    }

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    if (world.rank() == 0) {
      auto output_ptr = reinterpret_cast<DataType*>(taskData->outputs[0]);
      std::copy(C.begin(), C.end(), output_ptr);
    }
    return true;
  }

 private:
  std::shared_ptr<ppc::core::TaskData> taskData;
  boost::mpi::communicator world;
  std::vector<DataType> A;
  std::vector<DataType> B;
  std::vector<DataType> C;
  size_t m = 0;
  size_t k = 0;
  size_t n = 0;
};
}  // namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi
