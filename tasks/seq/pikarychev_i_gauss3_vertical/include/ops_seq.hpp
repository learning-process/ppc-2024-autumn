#pragma once

#include <algorithm>
#include <array>
#include <ostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace pikarychev_i_gauss3_vertical_seq {

using Kernel3x3 = std::array<double, 9>;
struct Image {
  struct Pixel {
    uint8_t r, g, b;

    bool operator==(const Pixel& rhs) const { return r == rhs.r && g == rhs.g && b == rhs.b; }
    Pixel operator+(const Pixel& rhs) const {
      return {.r = std::clamp<uint8_t>(r + rhs.r, 0, 255),
              .g = std::clamp<uint8_t>(g + rhs.g, 0, 255),
              .b = std::clamp<uint8_t>(b + rhs.b, 0, 255)};
    }
    Pixel operator*(double d) const {
      return {.r = std::clamp<uint8_t>(r * d, 0, 255),
              .g = std::clamp<uint8_t>(g * d, 0, 255),
              .b = std::clamp<uint8_t>(b * d, 0, 255)};
    }

    friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
      os << "{" << static_cast<int>(p.r) << "," << static_cast<int>(p.g) << "," << static_cast<int>(p.b) << "}";
      return os;
    }
  };

  size_t width;
  size_t height;
  std::vector<Pixel> data;

  void apply(const Kernel3x3& k, Image& out) const {
    for (size_t y = 1; y < height - 1; y++) {
      for (size_t x = 1; x < width - 1; x++) {
        // clang-format off
        out.data[((y - 1) * (width - 2)) + (x - 1)] = data[(y - 1) * width + (x - 1)] * k[0]
                                                     + data[(y + 0) * width + (x - 1)] * k[1]
                                                     + data[(y + 1) * width + (x - 1)] * k[2]
                                                     + data[(y - 1) * width + (x + 0)] * k[3]
                                                     + data[(y + 0) * width + (x + 0)] * k[4]
                                                     + data[(y + 1) * width + (x + 0)] * k[5]
                                                     + data[(y - 1) * width + (x + 1)] * k[6]
                                                     + data[(y + 0) * width + (x + 1)] * k[7]
                                                     + data[(y + 1) * width + (x + 1)] * k[8];
        // clang-format off
      }
    }
  }

  bool operator==(const Image& rhs) const { return width == rhs.width && height == rhs.height && data == rhs.data; }

  friend std::ostream& operator<<(std::ostream& os, const Image& img) {
    os << "[" << img.width << ":" << img.height << "]{ ";
    for (const auto& pixel : img.data) {
      os << pixel << ", ";
    }
    os << "}";
    return os;
  }

  static Image pad(const Image& img, size_t padding) {
    auto res = alloc(img.width + padding, img.height + padding);

    for (size_t y = 0; y < img.height; y++) {
      std::copy(img.data.begin() + (y * img.width), img.data.begin() + ((y + 1) * img.width),
                res.data.begin() + ((y + 1) * (img.width + 2) + 1));
    }

    return res;
  }

  static Image alloc(size_t w, size_t h) { return {w, h, std::vector<Pixel>(w * h)}; }
};

class TaskSeq : public ppc::core::Task {
 public:
  explicit TaskSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Kernel3x3 kernel{};
  Image imgin{};
  Image imgout{};

  static constexpr size_t padding = 2;
};

}  // namespace pikarychev_i_gauss3_vertical_seq