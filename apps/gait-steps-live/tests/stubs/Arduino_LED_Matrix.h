#pragma once
#include "Arduino.h"
// A fake 13 x 8 LED matrix that remembers the last frame drawn.
class Arduino_LED_Matrix {
 public:
  uint8_t last[104] = {0}; int draws = 0; bool begun = false; int bits = 0;
  void begin() { begun = true; }
  void setGrayscaleBits(int b) { bits = b; }
  void draw(uint8_t *frame) { memcpy(last, frame, 104); draws++; }
};
