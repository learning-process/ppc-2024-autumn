#include "mpi/oturin_a_image_smoothing/include/ops_mpi.hpp"

bool oturin_a_image_smoothing_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check elements count in i/o
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  width = (size_t)(taskData->inputs_count[0]);
  height = (size_t)(taskData->inputs_count[1]);
  input = std::vector<uint8_t>(width * height * 3);
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  input = std::vector<uint8_t>(tmp_ptr, tmp_ptr + width * height * 3);
  // Init values for output
  result = std::vector<uint8_t>(width * height * 3);
  CreateKernel();
  return true;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      SmoothPixel(x, y);
    }
  }
  return true;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  delete[] kernel;
  auto* result_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

// must be used before image processing
void oturin_a_image_smoothing_mpi::TestMPITaskSequential::CreateKernel() {
  int size = 2 * radius + 1;
  kernel = new float[size * size]{0};
  float sigma = 1.5;
  float norm = 0;

  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      kernel[(i + radius) * size + j + radius] = (float)(std::exp(-(i * i + j * j) / (2 * sigma * sigma)));
      norm += kernel[(i + radius) * size + j + radius];
    }
  }

  for (int i = 0; i < size * size; i++) {
    kernel[i] /= norm;
  }
}

void oturin_a_image_smoothing_mpi::TestMPITaskSequential::SmoothPixel(int x, int y) {
  int stride = width * 3;
  size_t sizek = 2 * radius + 1;
  float outR = 0.0f;
  float outG = 0.0f;
  float outB = 0.0f;
  for (int ry = -radius; ry <= radius; ry++) {
    for (int rx = -radius; rx <= radius; rx++) {
      int idX = clamp(x + rx, 0, width - 1);
      int idY = clamp(y + ry, 0, height - 1);
      int pos = idY * stride + idX * 3;
      int kernelPos = (ry + radius) * sizek + rx + radius;

      outR += input[pos] * kernel[kernelPos];
      outG += input[pos + 1] * kernel[kernelPos];
      outB += input[pos + 2] * kernel[kernelPos];
    }
  }
  int pos = y * stride + x * 3;
  result[pos] = (uint8_t)outR;
  result[pos + 1] = (uint8_t)outG;
  result[pos + 2] = (uint8_t)outB;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  // Check elements count in i/o
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
  }
  return true;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Init vectors
  if (world.rank() == 0) {
    width = taskData->inputs_count[0];
    height = taskData->inputs_count[1];
    // input = std::vector<uint8_t>(width * height * 3);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    input = std::vector<uint8_t>(tmp_ptr, tmp_ptr + width * height * 3);
    // Init values for output
    result = std::vector<uint8_t>(width * height * 3);
  }
  CreateKernel();
  return true;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  constexpr int TAG_EXIT = 0;
  constexpr int TAG_INFO = 2;
  constexpr int TAG_DATA = 3;
  constexpr int TAG_RESULT = 5;

#if defined(_MSC_VER) && !defined(__clang__)
  if (world.size() == 1) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        SmoothPixel(&result.data()[y * width * 3 + x * 3], x, y);
      }
    }
    return true;
  }
#endif

  if (world.rank() == 0) {
    int satellites = world.size() - 1;
    int escape = 0;
    int noescape = 1;

    for (int i = 1; i <= satellites; i++)  // send width
      world.send(i, TAG_INFO, &width, 1);

    int row = 0;
    while (row < height - 2) {
      for (int i = 0; i < std::min(satellites, height - 2 - row); i++) {
        world.send(i + 1, TAG_EXIT, &noescape, 1);
        world.send(i + 1, TAG_DATA, &input[(row + i) * width * 3], width * 3 * 3);
      }
      for (int i = 0; i < std::min(satellites, height - 2 - row); i++) {
        world.recv(i + 1, TAG_RESULT, &result[(row + i + 1) * width * 3], width * 3);
      }
      row += satellites;
    }
    for (int i = 1; i <= satellites; i++)  // close all satellite processes
      world.send(i, TAG_EXIT, &escape, 1);

    for (int x = 0; x < width; x++) {  // calculate bottom row
      SmoothPixel(&result[x * 3], x, 0);
    }
    for (int x = 0; x < width; x++) {  // calculate top row
      SmoothPixel(&result[(height - 1) * width * 3 + x * 3], x, height - 1);
    }
  } else {
    world.recv(0, TAG_INFO, &width, 1);
    input = std::vector<uint8_t>(width * 3 * 3);  // 3 RGB rows
    result = std::vector<uint8_t>(width * 3);     // 1 RGB row
    int escape = 0;
    height = INT_MAX;
    while (true) {
      world.recv(0, TAG_EXIT, &escape, 1);
      if (escape == 0) break;
      world.recv(0, TAG_DATA, input.data(), width * 3 * 3);

      for (int x = 0; x < width; x++) {
        SmoothPixel(&result[x * 3], x, 1);
      }

      world.send(0, TAG_RESULT, result.data(), width * 3);
    }
  }

  return true;
}

bool oturin_a_image_smoothing_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  delete[] kernel;
  if (world.rank() == 0) {
    auto* result_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), result_ptr);
  }
  return true;
}

// must be used before image processing
void oturin_a_image_smoothing_mpi::TestMPITaskParallel::CreateKernel() {
  int size = 2 * radius + 1;
  kernel = new float[size * size]{0};
  float sigma = 1.5;
  float norm = 0;

  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      kernel[(i + radius) * size + j + radius] = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
      norm += kernel[(i + radius) * size + j + radius];
    }
  }

  for (int i = 0; i < size * size; i++) {
    kernel[i] /= norm;
  }
}

void oturin_a_image_smoothing_mpi::TestMPITaskParallel::SmoothPixel(uint8_t* out, int x, int y) {
  int stride = width * 3;
  size_t sizek = 2 * radius + 1;
  float outR = 0.0f;
  float outG = 0.0f;
  float outB = 0.0f;
  for (int ry = -radius; ry <= radius; ry++) {
    for (int rx = -radius; rx <= radius; rx++) {
      int idX = clamp(x + rx, 0, width - 1);
      int idY = clamp(y + ry, 0, height - 1);
      int pos = idY * stride + idX * 3;
      int kernelPos = (ry + radius) * sizek + rx + radius;

      outR += input[pos] * kernel[kernelPos];
      outG += input[pos + 1] * kernel[kernelPos];
      outB += input[pos + 2] * kernel[kernelPos];
    }
  }
  out[0] = (uint8_t)outR;
  out[1] = (uint8_t)outG;
  out[2] = (uint8_t)outB;
}

#if defined(_WIN32) || defined(WIN32)
#else
oturin_a_image_smoothing_mpi::errno_t oturin_a_image_smoothing_mpi::fopen_s(FILE** f, const char* name,
                                                                            const char* mode) {
  errno_t ret = 0;
  assert(f);
  *f = fopen(name, mode);
  if (f == nullptr) ret = errno;
  return ret;
}
#endif

std::vector<uint8_t> oturin_a_image_smoothing_mpi::ReadBMP(const char* filename, int& w, int& h) {
  int i;
  FILE* f;
  fopen_s(&f, filename, "rb");
  if (f == nullptr) throw "Argument Exception";

  unsigned char info[54];
  size_t rc;
  rc = fread(info, sizeof(unsigned char), 54, f);  // read the 54-byte header
  if (rc == 0) {
    fclose(f);
    return std::vector<uint8_t>(0);
  }

  // extract image height and width from header
  int width = *(int*)&info[18];
  int height = *(int*)&info[22];

  // allocate 3 bytes per pixel
  int size = 3 * width * height;
  std::vector<uint8_t> data(size);

  unsigned char padding[3] = {0, 0, 0};
  size_t widthInBytes = (size_t)(width * BYTES_PER_PIXEL);
  size_t paddingSize = (4 - (widthInBytes) % 4) % 4;

  for (i = 0; i < height; i++) {
    rc = fread(data.data() + (i * widthInBytes), BYTES_PER_PIXEL, width, f);
    if (rc != width) break;
    rc = fread(padding, 1, paddingSize, f);
    if (rc != paddingSize) break;
  }
  fclose(f);
  w = width;
  h = height;

  return data;
}

int oturin_a_image_smoothing_mpi::clamp(int n, int lo, int hi) { return std::min(std::max(n, lo), hi); }
