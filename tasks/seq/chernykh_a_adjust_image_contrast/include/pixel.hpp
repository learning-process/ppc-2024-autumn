#pragma once

#include <iostream>
#include <vector>

namespace chernykh_a_adjust_image_contrast_seq {

class Pixel {
 public:
  Pixel() = default;
  explicit Pixel(uint8_t red, uint8_t green, uint8_t blue) : r(red), g(green), b(blue) {};

  static Pixel from_hex_color(uint32_t hex_color);

  Pixel with_contrast(float factor) const;
  bool operator==(const Pixel& other) const;
  friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel);

 private:
  uint8_t r{}, g{}, b{};
};

std::vector<Pixel> hex_colors_to_pixels(const std::vector<uint32_t>& hex_colors);

}  // namespace chernykh_a_adjust_image_contrast_seq
