#include "mpi/plekhanov_d_verticalgaus/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <numbers>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace plekhanov_d_verticalgaus_mpi {

void copy_input_data(double* source, std::vector<double>& destination, size_t size) {
  destination.resize(size);
  std::copy(source, source + size, destination.begin());
}

bool validate_image(const std::vector<double>& image, size_t width, size_t height) {
  const size_t img_size = width * height;
  for (size_t i = 0; i < img_size; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }
  return true;
}

void apply_gaussian_filter(const std::vector<double>& input, std::vector<double>& output, size_t width, size_t height) {
  const std::vector<double> gaussKernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
  const size_t padded_width = width + 2;

  // Pad the image
  std::vector<double> paddedImage(padded_width * height, 0);
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 1; j < width + 1; ++j) {
      paddedImage[i * padded_width + j] = input[i * width + j - 1];
    }
  }

  // Apply filter
  std::vector<double> filteredImageIntermediate(padded_width * height, 0);
  for (size_t i = 1; i < height - 1; ++i) {
    for (size_t j = 1; j < padded_width - 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += paddedImage[(i - l) * padded_width + j - k] * gaussKernel[(l + 1) * 3 + k + 1];
        }
      }
      filteredImageIntermediate[i * padded_width + j] = sum;
    }
  }

  // Strip padding
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 1; j < width + 1; ++j) {
      output[i * width + j - 1] = filteredImageIntermediate[i * padded_width + j];
    }
  }
}

}  // namespace plekhanov_d_verticalgaus_mpi

bool plekhanov_d_verticalgaus_mpi::VerticalGausSeqTest::pre_processing() {
  internal_order_test();
  inputWidth = taskData->inputs_count[1];
  inputHeight = taskData->inputs_count[2];
  inputImage.resize(taskData->inputs_count[0]);
  copy_input_data(reinterpret_cast<double*>(taskData->inputs[0]), inputImage, taskData->inputs_count[0]);
  outputImage.assign(inputWidth * inputHeight, 0);
  return true;
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausSeqTest::validation() {
  internal_order_test();
  inputImage.resize(taskData->inputs_count[0]);
  copy_input_data(reinterpret_cast<double*>(taskData->inputs[0]), inputImage, taskData->inputs_count[0]);

  size_t img_size = taskData->inputs_count[1] * taskData->inputs_count[2];

  return taskData->inputs_count[0] == img_size && taskData->outputs_count[0] == img_size &&
         taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3 &&
         validate_image(inputImage, inputWidth, inputHeight);
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausSeqTest::run() {
  internal_order_test();
  apply_gaussian_filter(inputImage, outputImage, inputWidth, inputHeight);
  return true;
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausSeqTest::post_processing() {
  internal_order_test();
  std::copy(outputImage.begin(), outputImage.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausMPITest::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    inputWidth = taskData->inputs_count[1];
    inputHeight = taskData->inputs_count[2];
    inputImage.resize(taskData->inputs_count[0]);
    copy_input_data(reinterpret_cast<double*>(taskData->inputs[0]), inputImage, taskData->inputs_count[0]);
    outputImage.assign(inputWidth * inputHeight, 0);
  }
  return true;
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausMPITest::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    inputImage.resize(taskData->inputs_count[0]);
    copy_input_data(reinterpret_cast<double*>(taskData->inputs[0]), inputImage, taskData->inputs_count[0]);

    size_t img_size = taskData->inputs_count[1] * taskData->inputs_count[2];

    return taskData->inputs_count[0] == img_size && taskData->outputs_count[0] == img_size &&
           taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3 &&
           validate_image(inputImage, inputWidth, inputHeight);
  }
  return true;
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausMPITest::run() {
  internal_order_test();

  broadcast(world, inputWidth, 0);
  broadcast(world, inputHeight, 0);

  std::vector<int> localImageWidths(world.size());
  int delta = inputWidth / world.size();

  if (inputWidth % world.size() != 0) {
    delta++;
  }
  if (world.rank() >= world.size() - world.size() * delta + inputWidth) {
    delta--;
  }

  delta += 2;
  boost::mpi::gather(world, delta, localImageWidths.data(), 0);

  std::vector<double> localImage(delta * inputHeight, 0);
  std::vector<double> sendImage(delta * inputHeight);

  if (world.size() == 1) {
    for (int i = 0; i < inputHeight; ++i) {
      for (int j = 1; j < inputWidth + 1; ++j) {
        localImage[i * (inputWidth + 2) + j] = inputImage[i * inputWidth + j - 1];
      }
    }
  } else {
    if (world.rank() == 0) {
      for (int i = 0; i < inputHeight; ++i) {
        for (int j = 0; j < delta - 1; ++j) {
          localImage[i * delta + j + 1] = inputImage[i * inputWidth + j];
        }
      }
      int gatheredImageIndex = delta - 2;
      for (int proc = 1; proc < world.size(); ++proc) {
        sendImage.assign(delta * inputHeight, 0);
        if (proc == world.size() - 1) {
          for (int i = 0; i < inputHeight; ++i) {
            for (int j = -1; j < localImageWidths[proc] - 2; ++j) {
              sendImage[i * localImageWidths[proc] + j + 1] = inputImage[i * inputWidth + j + gatheredImageIndex];
            }
          }
        } else {
          for (int i = 0; i < inputHeight; ++i) {
            for (int j = -1; j < localImageWidths[proc] - 1; ++j) {
              sendImage[i * localImageWidths[proc] + j + 1] = inputImage[i * inputWidth + j + gatheredImageIndex];
            }
          }
          gatheredImageIndex += localImageWidths[proc] - 2;
        }
        world.send(proc, 0, sendImage.data(), localImageWidths[proc] * inputHeight);
      }
    } else {
      world.recv(0, 0, localImage.data(), delta * inputHeight);
    }
  }

  std::vector<double> localFilteredImage(delta * inputHeight, 0);

  apply_gaussian_filter(localImage, localFilteredImage, delta, inputHeight);

  std::vector<double> filteredImageStripped((delta - 2) * inputHeight);

  for (int i = 0; i < inputHeight; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      filteredImageStripped[i * (delta - 2) + j - 1] = localFilteredImage[i * delta + j];
    }
  }

  std::vector<int> gatheredImageSizes(world.size());

  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); ++i) {
      gatheredImageSizes[i] = (localImageWidths[i] - 2) * inputHeight;
    }
  }

  std::vector<double> gatheredFilteredImage(inputWidth * inputHeight);
  boost::mpi::gatherv(world, filteredImageStripped, gatheredFilteredImage.data(), gatheredImageSizes, 0);

  if (world.rank() == 0) {
    int gatheredImageIndex = 0;
    for (int proc = 0; proc < world.size(); ++proc) {
      for (int i = 0; i < inputHeight; ++i) {
        for (int j = 0; j < localImageWidths[proc] - 2; ++j) {
          outputImage[i * inputWidth + j + gatheredImageIndex] =
              gatheredFilteredImage[i * (localImageWidths[proc] - 2) + j + gatheredImageIndex * inputHeight];
        }
      }
      gatheredImageIndex += localImageWidths[proc] - 2;
    }
  }

  return true;
}

bool plekhanov_d_verticalgaus_mpi::VerticalGausMPITest::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(outputImage.begin(), outputImage.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }
  return true;
}