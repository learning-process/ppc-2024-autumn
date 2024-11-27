#include "seq/oturin_a_image_smoothing/include/ops_seq.hpp"

// #include <algorithm>

bool oturin_a_image_smoothing_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check elements count in i/o
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool oturin_a_image_smoothing_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  width = (size_t)(taskData->inputs_count[0]);
  height = (size_t)(taskData->inputs_count[1]);
  input = std::vector<uint8_t>(width * height * 3);
  uint8_t* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  input = std::vector<uint8_t>(tmp_ptr, tmp_ptr + width * height * 3);
  // Init values for output
  result = std::vector<uint8_t>(width * height * 3);
  CreateKernel();
  return true;
}

bool oturin_a_image_smoothing_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      SmoothPixel(x, y);
    }
  }

  return true;
}

bool oturin_a_image_smoothing_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  delete[] kernel;
  uint8_t* result_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

// must be used before image processing
void oturin_a_image_smoothing_seq::TestTaskSequential::CreateKernel() {
  int size = 2 * radius + 1;
  kernel = new float[size * size];
  float sigma = 1.5;
  float norm = 0;

  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      kernel[(i + radius) * size + j + radius] = (float)(exp(-(i * i + j * j) / (2 * sigma * sigma)));
      norm += kernel[(i + radius) * size + j + radius];
    }
  }

  for (int i = 0; i < size * size; i++) {
    kernel[i] /= norm;
  }
}

void oturin_a_image_smoothing_seq::TestTaskSequential::SmoothPixel(int x, int y) {
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

#if defined(_WIN32) || defined(WIN32)
#else
oturin_a_image_smoothing_seq::errno_t oturin_a_image_smoothing_seq::fopen_s(FILE** f, const char* name,
                                                                            const char* mode) {
  errno_t ret = 0;
  assert(f);
  *f = fopen(name, mode);
  if (!*f) ret = errno;
  return ret;
}
#endif

std::vector<uint8_t> oturin_a_image_smoothing_seq::ReadBMP(const char* filename, int& w, int& h) {
  int i;
  FILE* f;
  fopen_s(&f, filename, "rb");
  if (f == 0) return std::vector<uint8_t>(0);

  if (f == NULL) throw "Argument Exception";

  unsigned char info[54];
  size_t rc;
  rc = fread(info, sizeof(unsigned char), 54, f);  // read the 54-byte header
  if (rc == 0) return std::vector<uint8_t>(0);

  // extract image height and width from header
  int width = *(int*)&info[18];
  int height = *(int*)&info[22];

  // allocate 3 bytes per pixel
  int size = 3 * width * height;
  std::vector<uint8_t> data(size);

  unsigned char padding[3] = {0, 0, 0};
  int widthInBytes = width * BYTES_PER_PIXEL;
  int paddingSize = (4 - (widthInBytes) % 4) % 4;
  // int stride = (widthInBytes) + paddingSize;

  for (i = 0; i < height; i++) {
    rc = fread(data.data() + (i * widthInBytes), BYTES_PER_PIXEL, width, f);
    rc = fread(padding, 1, paddingSize, f);
  }
  // if (rc == 0) return std::vector<uint8_t>(0);
  fclose(f);
  w = width;
  h = height;

  return data;
}

int oturin_a_image_smoothing_seq::clamp(int n, int lo, int hi) {
  if (n < lo)
    return lo;
  else if (n > hi)
    return hi;
  else
    return n;
}
