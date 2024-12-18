#include "seq/chernykh_a_adjust_image_contrast/include/pixel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

namespace chernykh_a_adjust_image_contrast_seq {

Pixel Pixel::from_hex_color(uint32_t hex_color) {
  return Pixel((hex_color >> 16) & 0xFF, (hex_color >> 8) & 0xFF, hex_color & 0xFF);
}

Pixel Pixel::with_contrast(float factor) const {
  auto adjust = [factor](uint8_t value) {
    auto new_value = 128 + int(std::round(factor * float(value - 128)));
    return uint8_t(std::clamp(new_value, 0, 255));
  };

  return Pixel(adjust(r), adjust(g), adjust(b));
}

bool Pixel::operator==(const Pixel& other) const { return r == other.r && g == other.g && b == other.b; }

std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
  auto format = [](uint8_t value) {
    auto ss = std::stringstream();
    ss << std::setw(2) << std::setfill('0') << std::hex << (int)value;
    return ss.str();
  };

  os << std::uppercase << "#" << format(pixel.r) << format(pixel.g) << format(pixel.b) << std::nouppercase;
  return os;
}

std::vector<Pixel> hex_colors_to_pixels(const std::vector<uint32_t>& hex_colors) {
  auto pixels = std::vector<Pixel>(hex_colors.size());
  std::transform(hex_colors.begin(), hex_colors.end(), pixels.begin(), Pixel::from_hex_color);
  return pixels;
}

}  // namespace chernykh_a_adjust_image_contrast_seq
