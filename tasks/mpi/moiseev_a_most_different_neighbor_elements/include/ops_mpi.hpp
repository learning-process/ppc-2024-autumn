#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>

#include "core/task/include/task.hpp"

namespace moiseev_a_most_different_neighbor_elements_mpi {

template <typename DataType>
struct Result {
  DataType diff;
  int64_t l_index;
  int64_t r_index;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & diff;
    ar & l_index;
    ar & r_index;
  }
};

template <typename DataType>
class MostDifferentNeighborElementsParallel : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElementsParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}

  bool pre_processing() override {
    internal_order_test();
    rank_ = world.rank();
    size_ = world.size();
    return true;
  }

  bool validation() override {
    internal_order_test();
    return world.rank() != 0 || (taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2);
  }

  bool run() override {
    internal_order_test();

    if (rank_ == 0) {
      total_size_ = taskData->inputs_count[0];
      auto* tmp_ptr = reinterpret_cast<DataType*>(taskData->inputs[0]);
      input_.assign(tmp_ptr, tmp_ptr + total_size_);
    }
    boost::mpi::broadcast(world, total_size_, 0);

    chunk_size_ = total_size_ / size_;
    remainder = total_size_ % size_;

    std::vector<int> sizes(world.size());
    std::vector<int> displs(world.size());
    std::vector<int> real_indicies(world.size());

    if (static_cast<int>(total_size_) < static_cast<int>(world.size())) {
      sizes[0] = total_size_;
    } else {
      for (int i = 0; i < world.size(); i++) {
        sizes[i] = chunk_size_ + (i < remainder ? 1 : 0);
        if (i > 0) {
          real_indicies[i] = real_indicies[i - 1] + sizes[i - 1] - 1;
          displs[i] = displs[i - 1] + sizes[i - 1] - 1;
        }
      }
    }

    int actual_chunk_size = sizes[world.rank()];
    displ = real_indicies[world.rank()];

    local_input_.resize(actual_chunk_size);
    boost::mpi::scatterv(world, input_.data(), sizes, displs, local_input_.data(), actual_chunk_size, 0);

    DataType local_max_diff = 0;
    int64_t local_l_index = 0;
    int64_t local_r_index = 1;

    if (!local_input_.empty()) {
      for (size_t i = 0; i < local_input_.size() - 1; ++i) {
        DataType diff = std::abs(local_input_[i] - local_input_[i + 1]);
        if (diff > local_max_diff) {
          local_max_diff = diff;
          local_l_index = static_cast<int64_t>(i);
          local_r_index = static_cast<int64_t>(i + 1);
        }
      }
    }

    int64_t global_l_index = local_l_index + displ;
    int64_t global_r_index = local_r_index + displ;

    Result<DataType> local_result = {local_max_diff, global_l_index, global_r_index};
    Result<DataType> global_result;

    boost::mpi::reduce(
        world, local_result, global_result,
        [](const auto& a, const auto& b) {
          return (a.diff > b.diff || (a.diff == b.diff && (a.l_index < b.l_index))) ? a : b;
        },
        0);

    if (rank_ == 0) {
      l_elem_index = global_result.l_index;
      r_elem_index = global_result.r_index;
    }
    return true;
  }

  bool post_processing() override {
    internal_order_test();

    if (rank_ == 0) {
      if (l_elem_index < input_.size() && r_elem_index < input_.size()) {
        reinterpret_cast<DataType*>(taskData->outputs[0])[0] = input_[l_elem_index];
        reinterpret_cast<DataType*>(taskData->outputs[0])[1] = input_[r_elem_index];
        reinterpret_cast<uint64_t*>(taskData->outputs[1])[0] = static_cast<uint64_t>(l_elem_index);
        reinterpret_cast<uint64_t*>(taskData->outputs[1])[1] = static_cast<uint64_t>(r_elem_index);
      }
    }
    return true;
  }

 private:
  std::vector<DataType> input_ = {};
  std::vector<DataType> local_input_ = {};
  boost::mpi::communicator world;
  size_t l_elem_index = 0;
  size_t r_elem_index = 0;
  size_t chunk_size_ = 0;
  size_t displ = 0;
  size_t total_size_ = 0;
  int remainder = 0;
  int rank_ = 0;
  int size_ = 0;
};

}  // namespace moiseev_a_most_different_neighbor_elements_mpi