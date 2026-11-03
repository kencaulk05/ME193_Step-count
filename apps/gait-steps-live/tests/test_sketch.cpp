#include "stubs/Arduino.h"
#include "stubs/Wire.h"
#include "stubs/Arduino_RouterBridge.h"
#include "stubs/Arduino_LED_Matrix.h"
uint32_t sim_us = 0; TwoWire Wire, Wire1; std::vector<Notification> notifications; BridgeClass Bridge;
#include "../sketch/sketch.ino"
#include <cmath>
#include <set>
#include <string>
#include <cstdlib>
static int failures_seen = 0;
#define CHECK(cond, ...) do { bool c_ = (cond); printf("  %-4s ", c_ ? "ok" : "FAIL"); printf(__VA_ARGS__); printf("\n"); if (!c_) failures_seen++; } while (0)
static void put16(uint8_t *p, int16_t v) { p[0] = (uint16_t)v >> 8; p[1] = (uint16_t)v & 0xFF; }
static void reset_world() { matrix = Arduino_LED_Matrix(); sim_us = 0; notifications.clear(); Wire = TwoWire(); Wire1 = TwoWire(); ok = false; counter = 0; failures = 0; bus = nullptr; addr = 0; snprintf(status, sizeof(status), "starting"); }
static void run(double seconds) { uint32_t end = sim_us + (uint32_t)(seconds * 1e6); while (sim_us < end) { loop(); sim_us += 20; } }   // each loop() pass costs ~20 us
static void still_motion(uint32_t, uint8_t *d) { put16(d + 0, 0); put16(d + 2, 0); put16(d + 4, 2048); put16(d + 8, 0); put16(d + 10, 0); put16(d + 12, 0); }

int main() {
  printf("1. sensor at 0x68 on the header pins: configuration, units and timing\n");
  reset_world(); Wire.dev.present = true; Wire.dev.addr = 0x68; Wire.dev.motion = [](uint32_t t, uint8_t *d) {
    put16(d + 0, 1024);            // ax = 0.5 g
    put16(d + 2, -2048);           // ay = -1.0 g
    put16(d + 4, 2048);            // az = +1.0 g
    put16(d + 8, 1640);            // gx = 100 dps
    put16(d + 10, -3280);          // gy = -200 dps
    put16(d + 12, 16);             // gz = 16/16.4 dps
  };
  setup(); run(2.0);
  CHECK(strstr(status, "0x68") && strstr(status, "Wire (SDA/SCL"), "status says: \"%s\"", status);
  auto reg = [&](uint8_t r) { for (int i = (int)Wire.dev.writes.size() - 1; i >= 0; i--) if (Wire.dev.writes[i].first == r) return (int)Wire.dev.writes[i].second; return -1; };
  CHECK(reg(0x6B) == 0x01, "PWR_MGMT_1 = 0x%02X (awake, gyro clock; a reset write came first: %s)", reg(0x6B), Wire.dev.writes.size() > 0 && Wire.dev.writes[0].second == 0x80 ? "yes" : "NO");
  CHECK(reg(0x19) == 9, "SMPLRT_DIV = %d (100 Hz)", reg(0x19));
  CHECK(reg(0x1A) == 3, "CONFIG = %d (44 Hz low-pass)", reg(0x1A));
  CHECK(reg(0x1B) == 0x18 && reg(0x1C) == 0x18, "gyro +-2000 dps (0x%02X) and accel +-16 g (0x%02X)", reg(0x1B), reg(0x1C));
  CHECK(Wire.clock == I2C_HZ, "I2C clock %u Hz", Wire.clock);
  size_t n = notifications.size();
  CHECK(n >= 198 && n <= 202, "%zu readings in 2 simulated seconds (expected 200)", n);
  bool consecutive = true; for (size_t i = 0; i < n; i++) if (notifications[i].args[0] != (double)i) consecutive = false;
  CHECK(consecutive, "the counter runs 0,1,2,... with no gaps");
  const auto &a = notifications[10].args;
  CHECK(fabs(a[1] - 0.5) < 1e-6 && fabs(a[2] + 1.0) < 1e-6 && fabs(a[3] - 1.0) < 1e-6, "accel in g: %.3f %.3f %.3f (expected 0.5 -1 1)", a[1], a[2], a[3]);
  CHECK(fabs(a[4] - 100.0) < 0.01 && fabs(a[5] + 200.0) < 0.01 && fabs(a[6] - 16 / 16.4) < 0.01, "gyro in dps: %.2f %.2f %.2f (expected 100 -200 0.98)", a[4], a[5], a[6]);
  double worst = 0; for (size_t i = 1; i < n; i++) worst = fmax(worst, fabs((notifications[i].at_us - notifications[i - 1].at_us) / 1000.0 - 10.0));
  CHECK(worst < 0.1, "spacing between readings: 10 ms +- %.3f ms", worst);
  CHECK(Bridge.status_fn && strstr(Bridge.status_fn().c_str(), "0x68"), "Python can ask imu_status and gets: \"%s\"", Bridge.status_fn ? Bridge.status_fn().c_str() : "(not registered)");

  printf("2. sensor at 0x69 (AD0 high)\n");
  reset_world(); Wire.dev.present = true; Wire.dev.addr = 0x69; Wire.dev.motion = still_motion; setup(); run(0.5);
  CHECK(strstr(status, "0x69") && notifications.size() > 40, "found at 0x69 and streaming (%zu readings)", notifications.size());

  printf("3. nothing connected, plugged in 3 s later\n");
  reset_world(); Wire.dev.addr = 0x68; Wire.dev.motion = still_motion; setup(); run(2.5);
  CHECK(strstr(status, "NO SENSOR FOUND") && notifications.empty(), "reports: \"%s\" and sends nothing", status);
  uint32_t plugged_at = sim_us; Wire.dev.present = true; run(3.0);
  double delay_s = notifications.empty() ? -1 : (notifications[0].at_us - plugged_at) / 1e6;
  CHECK(strstr(status, "0x68") && delay_s >= 0 && delay_s <= 1.4 && notifications.size() > 150, "after plugging in, streaming started %.2f s later (it retries once a second) and went on (%zu readings)", delay_s, notifications.size());

  printf("4. the sensor stops answering for a moment, then returns\n");
  reset_world(); Wire.dev.present = true; Wire.dev.motion = still_motion; setup(); run(1.0);
  size_t before = notifications.size(); Wire.dev.present = false; run(1.0);
  CHECK(notifications.size() == before && (strstr(status, "looking again") || strstr(status, "NO SENSOR")), "readings stop and the status says: \"%s\"", status);
  Wire.dev.present = true; run(2.5);
  CHECK(notifications.size() > before + 100, "it recovers by itself (%zu new readings)", notifications.size() - before);

  printf("5. a long stall (the board was busy for 1 s): no burst of catch-up readings\n");
  reset_world(); Wire.dev.present = true; Wire.dev.motion = still_motion; setup(); run(1.0);
  sim_us += 1000000; size_t b2 = notifications.size(); run(0.05);
  CHECK(notifications.size() - b2 <= 6, "%zu readings in the 50 ms after the stall (a rush would be 100)", notifications.size() - b2);
  run(1.0); double gaps = 0; for (size_t i = notifications.size() - 50; i < notifications.size(); i++) gaps = fmax(gaps, fabs((notifications[i].at_us - notifications[i - 1].at_us) / 1000.0 - 10.0));
  CHECK(gaps < 0.1, "and the spacing is back to 10 ms");

  printf("6. sensor on the Qwiic bus (built with TRY_QWIIC=%d)\n", TRY_QWIIC);
#if TRY_QWIIC
  reset_world(); Wire1.dev.present = true; Wire1.dev.motion = still_motion; setup(); run(0.5);
  CHECK(strstr(status, "Wire1 (Qwiic)") && notifications.size() > 40, "found on Wire1: \"%s\"", status);
#else
  reset_world(); Wire1.dev.present = true; Wire1.dev.motion = still_motion; setup(); run(0.5);
  CHECK(strstr(status, "NO SENSOR") && notifications.empty(), "with TRY_QWIIC=0 a sensor on Wire1 is NOT looked for (as intended): \"%s\"", status);
#endif

  printf("7. the LED matrix as a status display\n");
  reset_world(); Wire.dev.present = true; Wire.dev.motion = still_motion; setup();
  auto lit = [&]() { int n = 0; for (int i = 0; i < 104; i++) if (matrix.last[i]) n++; return n; };
  auto at = [&](int c, int r) { return matrix.last[r * 13 + c]; };
  CHECK(matrix.begun && matrix.bits == 8, "the matrix is started in 8-bit grayscale, as in the sumo app");
  CHECK(lit() == 1 && at(0, 0) == 120, "after setup it shows the idle dot (%d lit)", lit());
  CHECK(Bridge.led_fn != nullptr, "the led function is registered for Python");
  Bridge.led_fn(1, 5);  { bool row = true; for (int c = 0; c < 5; c++) row &= at(c, 3) == 255; CHECK(lit() == 5 && row && at(5, 3) == 0, "countdown with 5 left: 5 dots in the middle row (%d lit)", lit()); }
  Bridge.led_fn(2, 6);  { bool block = true; for (int c = 0; c < 6; c++) for (int r = 2; r < 6; r++) block &= at(c, r) == 255; CHECK(lit() == 24 && block && at(6, 3) == 0, "recording at 6 of 13: a 6 x 4 block (%d lit)", lit()); }
  Bridge.led_fn(2, 13); CHECK(lit() == 52, "recording complete: 13 x 4 (%d lit)", lit());
  Bridge.led_fn(2, 99); CHECK(lit() == 52, "a count above 13 is clamped (%d lit)", lit());
  Bridge.led_fn(2, -4); CHECK(lit() == 0, "a negative count shows nothing (%d lit)", lit());
  Bridge.led_fn(3, 0);  CHECK(lit() == 8 && at(2, 4) == 255 && at(9, 1) == 255, "saved: an 8-dot tick (%d lit)", lit());
  Bridge.led_fn(0, 0);  CHECK(lit() == 1 && at(0, 0) == 120, "idle again: one dim dot");
  Bridge.led_fn(77, 3); CHECK(lit() == 0, "an unknown mode shows an empty matrix instead of crashing");
  printf("8. the LED matrix shows the step count as a number (mode 4)\n");
  Bridge.led_fn(4, 0);   CHECK(lit() == 19 && at(5, 0) == 255 && at(7, 0) == 255 && at(4, 0) == 0 && at(8, 0) == 0, "0: a zero in the middle, its top arch in columns 5 to 7 (19 lit)");
  Bridge.led_fn(4, 1);   CHECK(lit() == 10, "1: (10 lit, got %d)", lit());
  Bridge.led_fn(4, 8);   CHECK(lit() == 17, "8: (17 lit, got %d)", lit());
  Bridge.led_fn(4, 7);   { bool top = true; for (int c = 4; c <= 8; c++) top &= at(c, 0) == 255; CHECK(lit() == 11 && top && at(3, 0) == 0 && at(9, 0) == 0, "7: one digit, centred in columns 4 to 8, with the bar across the top (11 lit)"); }
  Bridge.led_fn(4, 42);  { bool four = true, two = true; for (int c = 1; c <= 5; c++) four &= at(c, 4) == 255; for (int c = 7; c <= 11; c++) two &= at(c, 6) == 255; CHECK(lit() == 28 && four && two && at(6, 3) == 0, "42: a 4 in columns 1 to 5 and a 2 in columns 7 to 11, a column apart (28 lit)"); }
  Bridge.led_fn(4, 99);  CHECK(lit() == 30 && at(0, 7) == 0, "99: two nines, no marker line (30 lit)");
  Bridge.led_fn(4, 100); { bool line = true; for (int c = 0; c < 13; c++) line &= at(c, 7) == 255; CHECK(lit() == 43 && line, "100: a 99 with a line under it, meaning 99 or more (43 lit)"); }
  Bridge.led_fn(4, 12345); CHECK(lit() == 43, "12345: the same, not garbage (%d lit)", lit());
  Bridge.led_fn(4, -3);  CHECK(lit() == 19, "a negative number shows 0 (%d lit)", lit());
  { std::set<std::string> seen; bool all_lit = true;
    for (int n = 0; n <= 99; n++) { Bridge.led_fn(4, n); seen.insert(std::string((const char *)matrix.last, 104)); all_lit &= lit() > 0; }
    CHECK(seen.size() == 100 && all_lit, "every number from 0 to 99 draws a different picture (%zu different pictures)", seen.size()); }
  Bridge.led_fn(2, 6); CHECK(lit() == 24, "and the older modes still work after a number (recording at 6 of 13: %d lit)", lit());

  int draws0 = matrix.draws; run(1.0);
  CHECK(matrix.draws == draws0, "the 100 Hz IMU loop does not touch the matrix");
  CHECK(notifications.size() >= 98, "and the IMU stream is unaffected (%zu readings)", notifications.size());

  printf("\n%s (%d problem%s)\n", failures_seen ? "SOME FAILED" : "ALL PASSED", failures_seen, failures_seen == 1 ? "" : "s");
  return failures_seen ? 1 : 0;
}
