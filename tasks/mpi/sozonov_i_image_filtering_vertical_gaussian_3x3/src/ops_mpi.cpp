#include "mpi/sozonov_i_image_filtering_vertical_gaussian_3x3/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <numbers>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init image
  image = std::vector<double>(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], image.begin());
  width = taskData->inputs_count[1];
  height = taskData->inputs_count[2];
  // Init filtered image
  filtered_image = std::vector<double>(width * height, 0);
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Init image
  image = std::vector<double>(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], image.begin());

  size_t img_size = taskData->inputs_count[1] * taskData->inputs_count[2];

  // Check pixels range from 0 to 255
  for (size_t i = 0; i < img_size; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }

  // Check size of image
  return taskData->inputs_count[0] == img_size && taskData->outputs_count[0] == img_size &&
         taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
  std::vector<double> tmp_image((width + 2) * height, 0);
  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      tmp_image[i * (width + 2) + j] = image[i * width + j - 1];
    }
  }
  std::vector<double> tmp_filtered_image((width + 2) * height, 0);
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += tmp_image[(i - l) * (width + 2) + j - k] * kernel[(l + 1) * 3 + k + 1];
        }
      }
      tmp_filtered_image[i * (width + 2) + j] = sum;
    }
  }
  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      filtered_image[i * width + j - 1] = tmp_filtered_image[i * (width + 2) + j];
    }
  }
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto *out = reinterpret_cast<double *>(taskData->outputs[0]);
  std::copy(filtered_image.begin(), filtered_image.end(), out);
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init image
    image = std::vector<double>(taskData->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], image.begin());
    width = taskData->inputs_count[1];
    height = taskData->inputs_count[2];
    // Init filtered image
    filtered_image = std::vector<double>(width * height, 0);
  }
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init image
    image = std::vector<double>(taskData->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], image.begin());

    size_t img_size = taskData->inputs_count[1] * taskData->inputs_count[2];

    // Check pixels range from 0 to 255
    for (size_t i = 0; i < img_size; ++i) {
      if (image[i] < 0 || image[i] > 255) {
        return false;
      }
    }

    // Check size of image
    return taskData->inputs_count[0] == img_size && taskData->outputs_count[0] == img_size &&
           taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3;
  }
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, width, 0);
  broadcast(world, height, 0);

  std::vector<int> col_num(world.size());

  int delta = width / world.size();
  if (width % world.size() != 0) {
    delta++;
  }
  if (world.rank() >= world.size() - world.size() * delta + width) {
    delta--;
  }

  delta = delta + 2;

  boost::mpi::gather(world, delta, col_num.data(), 0);

  std::vector<double> local_image(delta * height, 0);
  std::vector<double> send_image(delta * height);

  if (world.size() == 1) {
    for (int i = 0; i < height; ++i) {
      for (int j = 1; j < width + 1; ++j) {
        local_image[i * (width + 2) + j] = image[i * width + j - 1];
      }
    }
  } else {
    if (world.rank() == 0) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < delta - 1; ++j) {
          local_image[i * delta + j + 1] = image[i * width + j];
        }
      }
      int idx = delta - 2;
      for (int proc = 1; proc < world.size(); ++proc) {
        send_image = std::vector<double>(delta * height, 0);
        if (proc == world.size() - 1) {
          for (int i = 0; i < height; ++i) {
            for (int j = -1; j < col_num[proc] - 2; ++j) {
              send_image[i * col_num[proc] + j + 1] = image[i * width + j + idx];
            }
          }
        } else {
          for (int i = 0; i < height; ++i) {
            for (int j = -1; j < col_num[proc] - 1; ++j) {
              send_image[i * col_num[proc] + j + 1] = image[i * width + j + idx];
            }
          }
          idx += col_num[proc] - 2;
        }
        world.send(proc, 0, send_image.data(), col_num[proc] * height);
      }
    } else {
      world.recv(0, 0, local_image.data(), delta * height);
    }
  }

  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

  std::vector<double> local_filtered_image(delta * height, 0);

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += local_image[(i - l) * delta + j - k] * kernel[(l + 1) * 3 + k + 1];
        }
      }
      local_filtered_image[i * delta + j] = sum;
    }
  }

  std::vector<double> back_image((delta - 2) * height);

  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      back_image[i * (delta - 2) + j - 1] = local_filtered_image[i * delta + j];
    }
  }

  std::vector<int> recv_counts(world.size());

  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); ++i) {
      recv_counts[i] = (col_num[i] - 2) * height;
    }
  }

  std::vector<double> gathered_image(width * height);
  boost::mpi::gatherv(world, back_image, gathered_image.data(), recv_counts, 0);

  if (world.rank() == 0) {
    int idx = 0;
    for (int proc = 0; proc < world.size(); ++proc) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < col_num[proc] - 2; ++j) {
          filtered_image[i * width + j + idx] = gathered_image[i * (col_num[proc] - 2) + j + idx * height];
        }
      }
      idx += col_num[proc] - 2;
    }
  }

  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *out = reinterpret_cast<double *>(taskData->outputs[0]);
    std::copy(filtered_image.begin(), filtered_image.end(), out);
  }
  return true;
}