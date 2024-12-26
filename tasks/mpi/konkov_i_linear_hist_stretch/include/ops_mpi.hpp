#ifndef MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP_
#define MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP_

#include <mpi.h>

namespace konkov_i_linear_hist_stretch {

class LinearHistogramStretch {
 public:
  explicit LinearHistogramStretch(int image_size, int* image_data);
  ~LinearHistogramStretch();
  bool validation() const;
  bool pre_processing();
  bool run();
  bool post_processing();

 private:
  int image_size_;
  int* image_data_;
  int rank_;
  int size_;
  int local_size_;
  int* local_data_;

  void distribute_data();
  void gather_data();
};

}  // namespace konkov_i_linear_hist_stretch

#endif  // MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP_