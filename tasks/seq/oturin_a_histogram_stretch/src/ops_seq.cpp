#include "seq/oturin_a_histogram_stretch/include/ops_seq.hpp"

bool oturin_a_histogram_stretch_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check elements count in i/o
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] > 0 &&
         taskData->outputs_count[1] > 0;
}

bool oturin_a_histogram_stretch_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  width = (size_t)(taskData->inputs_count[0]);
  height = (size_t)(taskData->inputs_count[1]);
  input = std::vector<uint8_t>(width * height * 3);
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  input = std::vector<uint8_t>(tmp_ptr, tmp_ptr + width * height * 3);
  // Init values for output
  result = std::vector<uint8_t>(width * height * 3);
  return true;
}

bool oturin_a_histogram_stretch_seq::TestTaskSequential::run() {
  internal_order_test();

  uint8_t minimum = 255;
  uint8_t maximum = 0;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {  // read only one channel (task requires grayscale image)
      minimum = std::min(minimum, input[y * width * 3 + x * 3]);
      maximum = std::max(maximum, input[y * width * 3 + x * 3]);
    }
  }

  if (maximum == minimum) {
    for (int i = 0; i < height * width * 3; i++) result[i] = input[i];  // histogram doesn't need to stretch
  } else {
    float pixMultiplier = 255.0f / (maximum - minimum);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width * 3; x++) {
        result[y * width * 3 + x] = std::round((input[y * width * 3 + x] - minimum) * pixMultiplier);
      }
    }
  }

  return true;
}

bool oturin_a_histogram_stretch_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* result_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

#if defined(_WIN32) || defined(WIN32)
#else
oturin_a_histogram_stretch_seq::errno_t oturin_a_histogram_stretch_seq::fopen_s(FILE** f, const char* name,
                                                                                const char* mode) {
  errno_t ret = 0;
  assert(f);
  *f = fopen(name, mode);
  if (f == nullptr) ret = errno;
  return ret;
}
#endif

// based on https://stackoverflow.com/questions/9296059
std::vector<uint8_t> oturin_a_histogram_stretch_seq::ReadBMP(const char* filename, int& w, int& h) {
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
  size_t widthInBytes = width * BYTES_PER_PIXEL;
  size_t paddingSize = (4 - (widthInBytes) % 4) % 4;

  for (i = 0; i < height; i++) {
    rc = fread(data.data() + (i * widthInBytes), BYTES_PER_PIXEL, width, f);
    if (rc != (size_t)width) break;
    rc = fread(padding, 1, paddingSize, f);
    if (rc != paddingSize) break;
  }
  fclose(f);
  w = width;
  h = height;

  return data;
}
