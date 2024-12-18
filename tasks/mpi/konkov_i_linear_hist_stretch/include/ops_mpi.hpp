#ifndef MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP_
#define MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP_

#include <mpi.h>

#include <vector>

namespace konkov_i_linear_hist_stretch {

class LinearHistogramStretch {
 public:
  explicit LinearHistogramStretch(int image_size, int* image_data);
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
  int global_min_;
  int global_max_;

  void calculate_local_min_max();
  void stretch_pixels();
};

}  // namespace konkov_i_linear_hist_stretch

#endif  // MODULES_TASK_2_KONKOV_I_LINEAR_HIST_STRETCH_OPS_MPI_HPP_
