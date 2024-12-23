#include "mpi/oturin_a_histogram_stretch/include/ops_mpi.hpp"

bool oturin_a_histogram_stretch_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check elements count in i/o
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] > 0 &&
         taskData->outputs_count[1] > 0;
}

bool oturin_a_histogram_stretch_mpi::TestMPITaskSequential::pre_processing() {
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

bool oturin_a_histogram_stretch_mpi::TestMPITaskSequential::run() {
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

bool oturin_a_histogram_stretch_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* result_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

bool oturin_a_histogram_stretch_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  // Check elements count in i/o
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] > 0 &&
           taskData->outputs_count[1] > 0;
  }
  return true;
}

bool oturin_a_histogram_stretch_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Init vectors
  if (world.rank() == 0) {
    width = taskData->inputs_count[0];
    height = taskData->inputs_count[1];
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    input = std::vector<uint8_t>(tmp_ptr, tmp_ptr + width * height * 3);
    // Init values for output
    result = std::vector<uint8_t>(width * height * 3);
  }
  return true;
}

bool oturin_a_histogram_stretch_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  constexpr int TAG_PICSIZE = 1;
  constexpr int TAG_DATA = 2;
  constexpr int TAG_MINIMUM = 3;
  constexpr int TAG_MAXIMUM = 4;
  constexpr int TAG_GMINIMUM = 5;
  constexpr int TAG_GMAXIMUM = 6;
  constexpr int TAG_RESULT = 7;

  uint8_t minimum = 255;
  uint8_t maximum = 0;

  float pixMultiplier = -1;

  int picMemSize = 0;

  if (world.rank() == 0) {
    int workWorldSize = world.size();
    picMemSize = width * height;

    int blockSize = picMemSize / world.size();
    int lastBlock = picMemSize - blockSize * (world.size() - 1);
    if (blockSize == 0) workWorldSize = 1;

    blockSize *= 3;
    lastBlock *= 3;

    for (int i = 1; i < world.size(); i++) world.send(i, TAG_PICSIZE, &blockSize, 1);

    for (int i = 1; i < workWorldSize; i++) world.send(i, TAG_DATA, &input[(i - 1) * blockSize], blockSize);

    for (int i = 0; i < lastBlock; i += 3) {
      minimum = std::min(minimum, input[blockSize * (workWorldSize - 1) + i]);
      maximum = std::max(maximum, input[blockSize * (workWorldSize - 1) + i]);
    }

    for (int i = 1; i < workWorldSize; i++) {
      uint8_t temp;
      world.recv(i, TAG_MINIMUM, &temp, 1);
      minimum = std::min(minimum, temp);
      world.recv(i, TAG_MAXIMUM, &temp, 1);
      maximum = std::max(maximum, temp);
    }

    for (int i = 1; i < workWorldSize; i++) {
      world.send(i, TAG_GMINIMUM, &minimum, 1);
      world.send(i, TAG_GMAXIMUM, &maximum, 1);
    }

    if (maximum == minimum) {
      result = input;  // histogram doesn't need to stretch
    } else {
      pixMultiplier = 255.0f / (maximum - minimum);

      for (int i = 0; i < lastBlock; i++) {
        result[blockSize * (workWorldSize - 1) + i] =
            (uint8_t)std::round((input[blockSize * (workWorldSize - 1) + i] - minimum) * pixMultiplier);
      }

      for (int i = 1; i < workWorldSize; i++) {
        world.recv(i, TAG_RESULT, &result[(i - 1) * blockSize], blockSize);
      }
    }
  } else {
    world.recv(0, TAG_PICSIZE, &picMemSize, 1);

    if (picMemSize == 0) {
      return true;
    }

    std::vector<uint8_t> data(picMemSize);
    world.recv(0, TAG_DATA, data.data(), picMemSize);

    for (int i = 0; i < picMemSize; i += 3) {
      minimum = std::min(minimum, data[i]);
      maximum = std::max(maximum, data[i]);
    }
    world.send(0, TAG_MINIMUM, &minimum, 1);
    world.send(0, TAG_MAXIMUM, &maximum, 1);

    world.recv(0, TAG_GMINIMUM, &minimum, 1);
    world.recv(0, TAG_GMAXIMUM, &maximum, 1);

    if (maximum != minimum) {
      pixMultiplier = 255.0f / (maximum - minimum);

      for (int i = 0; i < picMemSize; i++) {
        data[i] = (uint8_t)std::round((data[i] - minimum) * pixMultiplier);
      }

      world.send(0, TAG_RESULT, data.data(), picMemSize);
    }
  }

  return true;
}

bool oturin_a_histogram_stretch_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* result_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), result_ptr);
  }
  return true;
}

#if defined(_WIN32) || defined(WIN32)
#else
oturin_a_histogram_stretch_mpi::errno_t oturin_a_histogram_stretch_mpi::fopen_s(FILE** f, const char* name,
                                                                                const char* mode) {
  errno_t ret = 0;
  assert(f);
  *f = fopen(name, mode);
  if (f == nullptr) ret = errno;
  return ret;
}
#endif

// based on https://stackoverflow.com/questions/9296059
std::vector<uint8_t> oturin_a_histogram_stretch_mpi::ReadBMP(const char* filename, int& w, int& h) {
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
