#pragma once
#include "Arduino.h"
#include <functional>
#include <vector>
// A fake I2C bus with at most one MPU-6050-like device on it.
struct FakeDevice {
  bool present = false; uint8_t addr = 0x68; uint8_t who = 0x68;
  uint8_t regs[256] = {0};
  std::function<void(uint32_t, uint8_t *)> motion;        // fills the 14 data bytes (0x3B..0x48) for the given time
  std::vector<std::pair<uint8_t, uint8_t>> writes;        // every register write, in order
};
class TwoWire {
 public:
  FakeDevice dev; uint8_t cur = 0, pointer = 0; std::vector<uint8_t> tx; uint8_t rx[32]; uint8_t rxi = 0; int begun = 0; uint32_t clock = 0;
  void begin() { begun++; }
  void setClock(uint32_t hz) { clock = hz; }
  void beginTransmission(uint8_t a) { cur = a; tx.clear(); }
  size_t write(uint8_t v) { tx.push_back(v); return 1; }
  uint8_t endTransmission(bool = true) {
    if (!dev.present || cur != dev.addr) return 2;          // nobody answered
    if (tx.size() >= 2) { dev.regs[tx[0]] = tx[1]; dev.writes.push_back({tx[0], tx[1]}); pointer = tx[0]; if (tx[0] == 0x6B && tx[1] == 0x80) dev.regs[0x6B] = 0; }
    else if (tx.size() == 1) pointer = tx[0];
    return 0;
  }
  size_t requestFrom(uint8_t a, uint8_t n) {
    if (!dev.present || a != dev.addr) return 0;
    uint8_t data[14] = {0};
    if (dev.motion) dev.motion(sim_us, data);
    for (uint8_t i = 0; i < n; i++) {
      uint8_t r = pointer + i;
      rx[i] = (r == 0x75) ? dev.who : (r >= 0x3B && r < 0x3B + 14) ? data[r - 0x3B] : dev.regs[r];
    }
    rxi = 0; return n;
  }
  int read() { return rx[rxi++]; }
};
extern TwoWire Wire, Wire1;
