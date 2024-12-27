#include "mpi\sharamygina_i_horizontal_line_filtraiton\include\ops_mpi.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>

bool sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  if (world.rank() == 0) {
    auto* input_buffer = reinterpret_cast<unsigned char*>(taskData->inputs[0]);
    original_data_.assign(input_buffer, input_buffer + rows_ * cols_);
  }

  boost::mpi::broadcast(world, rows_, 0);
  boost::mpi::broadcast(world, cols_, 0);

  return true;
}

bool sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (!taskData || taskData->inputs.size() < 9 || taskData->inputs_count.size() < 2 || taskData->outputs.empty() ||
        taskData->outputs_count.empty()) {
      return false;
    }
  }

  int exp_size = taskData->inputs_count[0] * taskData->inputs_count[1];
  if (taskData->inputs_count[0] < 3 || taskData->inputs_count[1] < 3 || taskData->inputs_count[0] < world.size() ||
      taskData->inputs.size() != exp_size || taskData->outputs.size() != exp_size) {
    return false;
  }
  return true;
}

bool sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi::run() {
  internal_order_test();
  MPI_Status status;
  int count_of_proc = world.size();
  int myrank = world.rank();
  result_data_.resize(rows_ * cols_);
  int block_on_proc = rows_ / count_of_proc, remainder = rows_ % count_of_proc;
  std::vector<unsigned char> temporaryImage(block_on_proc * cols_ + 2 * cols_);
  std::vector<unsigned char> local_data(block_on_proc * cols_);

  if (myrank == 0) {
    if (count_of_proc != 1) {
      temporaryImage.resize((block_on_proc + remainder + 1) * cols_);
      for (int i = 0; i < (block_on_proc + remainder + 1) * cols_; i++) temporaryImage[i] = original_data_[i];
    }
    for (int i = 1; i < count_of_proc; i++) {
      int count = block_on_proc * cols_ + cols_;
      if (i != count_of_proc - 1) count += cols_;
      MPI_Send(original_data_.data() + (i * block_on_proc * cols_) + (remainder - 1) * cols_, count, MPI_UNSIGNED_CHAR,
               i, 0, MPI_COMM_WORLD);
    }
  } else {
    if (myrank != count_of_proc - 1) {
      MPI_Recv(&temporaryImage[0], (block_on_proc + 2) * cols_ + 2, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    } else {
      temporaryImage.resize((block_on_proc + 1) * cols_);
      MPI_Recv(temporaryImage.data(), (block_on_proc + 1) * cols_, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  if (myrank == count_of_proc - 1 && count_of_proc != 1) {
    for (int i = 0; i < block_on_proc; i++)
      for (int j = 0; j < cols_; j++)
        local_data[i * cols_ + j] = InputAnotherPixel(temporaryImage, i + 1, j, block_on_proc + 1, cols_);
  } else if (myrank != 0) {
    for (int i = 1; i < block_on_proc + 1; i++)
      for (int j = 0; j < cols_; j++)
        local_data[(i - 1) * cols_ + j] = InputAnotherPixel(temporaryImage, i, j, block_on_proc + 2, cols_);
  } else {
    if (count_of_proc != 1) {
      for (int i = 0; i < block_on_proc + remainder; i++)
        for (int j = 0; j < cols_; j++)
          result_data_[i * cols_ + j] = InputAnotherPixel(temporaryImage, i, j, block_on_proc + remainder + 1, cols_);
    } else {
      for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++)
          result_data_[i * cols_ + j] = InputAnotherPixel(original_data_, i, j, rows_, cols_);
    }
  }

  if (myrank != 0) {
    MPI_Send(local_data.data(), block_on_proc * cols_, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
  } else {
    for (int i = 1; i < count_of_proc; i++)
      MPI_Recv(result_data_.data() + ((block_on_proc + remainder) * cols_) + ((i - 1) * block_on_proc * cols_),
               block_on_proc * cols_, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &status);
  }
  return true;
}

bool sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    unsigned int* output_ptr = reinterpret_cast<unsigned int*>(taskData->outputs[0]);
    std::copy(result_data_.begin(), result_data_.end(), output_ptr);
  }
  return true;
}

unsigned int sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi::InputAnotherPixel(
    const std::vector<unsigned int>& image, int x, int y, int rows, int cols) {
  unsigned int sum = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      int tX = x + i - 1, tY = y + j - 1;
      if (tX < 0 || tX > rows_ - 1) tX = x;
      if (tY < 0 || tY > cols_ - 1) tY = y;
      if (tX * cols + tY >= cols * rows) {
        tX = x;
        tY = y;
      }
      sum += static_cast<unsigned int>(image[tX * cols + tY] * (gauss[i][j]));
    }
  return sum / 16;
}
