// How many steps? - the live counter - UNO Q microcontroller side: the IMU stream, plus the LED matrix as a display for the step count.
//
// Reads an MPU-6050 over I2C at 100 Hz and sends EVERY reading to Python:
//     Bridge.notify("imu", n, ax, ay, az, gx, gy, gz)      n = sample counter, a* in g, g* in degrees per second
// Python can call:
//     imu_status()      one line saying what sensor was found
//     led(mode, n)      mode 0 = idle (one dot), 1 = countdown (n dots), 2 = recording (progress bar, n of 13 columns), 3 = saved (tick),
//                       4 = a NUMBER: the step count n, one or two digits (99 or more shows 99 with a line under it)
//
// WIRING (3.3 V logic: power the module from 3V3, NOT 5V):  VCC -> 3V3,  GND -> GND,  SDA -> SDA,  SCL -> SCL.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

#ifndef TRY_QWIIC
#define TRY_QWIIC 0          // 1 = if nothing answers on the header pins, also look on the Qwiic connector (Wire1)
#endif
#ifndef I2C_HZ
#define I2C_HZ 100000        // 100 kHz is the safe choice for a longer cable; 400000 is fine on short wires
#endif

const uint8_t  REG_WHO_AM_I = 0x75, REG_PWR_MGMT_1 = 0x6B, REG_SMPLRT_DIV = 0x19, REG_CONFIG = 0x1A,
               REG_GYRO_CONFIG = 0x1B, REG_ACCEL_CONFIG = 0x1C, REG_ACCEL_XOUT_H = 0x3B;
const uint32_t PERIOD_US = 10000;       // 100 samples per second
const float    ACCEL_LSB_PER_G   = 2048.0f;   // +-16 g range
const float    GYRO_LSB_PER_DPS  = 16.4f;     // +-2000 degrees/second range

Arduino_LED_Matrix matrix;
uint8_t  frame[104];                    // 13 columns x 8 rows
TwoWire *bus = nullptr;
uint8_t  addr = 0;
bool     ok = false;
char     status[96] = "starting";
uint32_t counter = 0;
uint32_t next_us = 0;
uint8_t  failures = 0;

// ---- plain I2C helpers -------------------------------------------------------------------------------------------
bool write_reg(TwoWire &w, uint8_t a, uint8_t reg, uint8_t value) {
  w.beginTransmission(a);
  w.write(reg);
  w.write(value);
  return w.endTransmission() == 0;
}

bool read_regs(TwoWire &w, uint8_t a, uint8_t reg, uint8_t *buf, uint8_t n) {
  w.beginTransmission(a);
  w.write(reg);
  if (w.endTransmission(false) != 0) return false;         // repeated start
  if (w.requestFrom(a, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = w.read();
  return true;
}

// ---- conversions -------------------------------------------------------------------------------------------------
int16_t be16(const uint8_t *p) { return (int16_t)(((uint16_t)p[0] << 8) | p[1]); }
float   raw_to_g(int16_t raw)   { return raw / ACCEL_LSB_PER_G; }
float   raw_to_dps(int16_t raw) { return raw / GYRO_LSB_PER_DPS; }

// ---- find and configure the sensor -------------------------------------------------------------------------------
bool try_device(TwoWire &w, const char *name, uint8_t a) {
  uint8_t who = 0;
  if (!read_regs(w, a, REG_WHO_AM_I, &who, 1)) return false;
  write_reg(w, a, REG_PWR_MGMT_1, 0x80);                               // reset
  delay(100);
  bool good = write_reg(w, a, REG_PWR_MGMT_1, 0x01)                    // wake up, clock from the gyro
           && write_reg(w, a, REG_SMPLRT_DIV, 9)                       // 1 kHz / (1 + 9) = 100 Hz
           && write_reg(w, a, REG_CONFIG, 3)                           // low-pass filter at about 44 Hz
           && write_reg(w, a, REG_GYRO_CONFIG, 0x18)                   // +-2000 dps
           && write_reg(w, a, REG_ACCEL_CONFIG, 0x18);                 // +-16 g
  if (!good) return false;
  delay(50);
  bus = &w;
  addr = a;
  snprintf(status, sizeof(status), "sensor found on %s at 0x%02X (WHO_AM_I = 0x%02X)", name, a, who);
  return true;
}

bool begin_imu() {
  Wire.begin();
  Wire.setClock(I2C_HZ);
  if (try_device(Wire, "Wire (SDA/SCL pins)", 0x68)) return true;
  if (try_device(Wire, "Wire (SDA/SCL pins)", 0x69)) return true;
#if TRY_QWIIC
  Wire1.begin();
  Wire1.setClock(I2C_HZ);
  if (try_device(Wire1, "Wire1 (Qwiic)", 0x68)) return true;
  if (try_device(Wire1, "Wire1 (Qwiic)", 0x69)) return true;
#endif
  snprintf(status, sizeof(status), "NO SENSOR FOUND: check the wiring (3V3, GND, SDA, SCL)");
  return false;
}

String imu_status() { return String(status); }

// ---- the LED matrix as a status display ----------------------------------------------------------------------------
void put(int col, int row, uint8_t value) {
  if (col >= 0 && col < 13 && row >= 0 && row < 8) frame[row * 13 + col] = value;
}

// A 5 x 7 digit font: each digit is 7 rows, each row 5 bits (bit 4 is the leftmost column).
const uint8_t DIGITS[10][7] = {
  {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},   // 0
  {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},   // 1
  {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},   // 2
  {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E},   // 3
  {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},   // 4
  {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},   // 5
  {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},   // 6
  {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},   // 7
  {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},   // 8
  {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},   // 9
};

void draw_digit(int digit, int col0) {
  for (int r = 0; r < 7; r++)
    for (int c = 0; c < 5; c++)
      if (DIGITS[digit][r] & (1 << (4 - c))) put(col0 + c, r, 255);
}

void draw_number(int n) {
  if (n < 0) n = 0;
  bool over = n > 99;
  if (over) n = 99;
  if (n >= 10) {
    draw_digit(n / 10, 1);                                             // two digits side by side, a column apart
    draw_digit(n % 10, 7);
  } else {
    draw_digit(n, 4);                                                  // one digit in the middle
  }
  if (over) for (int c = 0; c < 13; c++) put(c, 7, 255);               // 99 or more: a line under the 99
}

int set_led(int mode, int cols) {
  memset(frame, 0, sizeof(frame));
  if (mode != 4) {                                                     // for modes 1 and 2, cols is a width
    if (cols < 0) cols = 0;
    if (cols > 13) cols = 13;
  }
  if (mode == 0) {
    put(0, 0, 120);                                                    // idle: one dim dot, the app is alive
  } else if (mode == 1) {
    for (int c = 0; c < cols; c++) put(c, 3, 255);                     // countdown: one dot per second left
  } else if (mode == 2) {
    for (int c = 0; c < cols; c++) for (int r = 2; r < 6; r++) put(c, r, 255);   // recording: a bar that fills up
  } else if (mode == 4) {
    draw_number(cols);                                                 // the step count (here cols is the number, not a width)
  } else if (mode == 3) {
    const int tick[][2] = {{2, 4}, {3, 5}, {4, 6}, {5, 5}, {6, 4}, {7, 3}, {8, 2}, {9, 1}};
    for (auto &p : tick) put(p[0], p[1], 255);                         // saved: a tick
  }
  matrix.draw(frame);
  return mode;
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(8);
  set_led(0, 0);
  Bridge.begin();
  Bridge.provide_safe("imu_status", imu_status);
  Bridge.provide_safe("led", set_led);
  ok = begin_imu();
  next_us = micros() + PERIOD_US;
}

void loop() {
  if (!ok) {                                   // keep looking, once a second
    delay(1000);
    ok = begin_imu();
    next_us = micros() + PERIOD_US;
    return;
  }
  int32_t wait_us = (int32_t)(next_us - micros());
  if (wait_us > 1500) { delay(1); return; }    // not time yet: let the other tasks run
  if (wait_us > 0) delayMicroseconds(wait_us);
  next_us += PERIOD_US;
  if ((int32_t)(micros() - next_us) > (int32_t)(5 * PERIOD_US)) next_us = micros() + PERIOD_US;   // after a long stall: do not rush

  uint8_t b[14];                               // accel x,y,z, temperature, gyro x,y,z (each 16 bits, high byte first)
  if (!read_regs(*bus, addr, REG_ACCEL_XOUT_H, b, 14)) {
    if (++failures >= 20) {                    // the sensor stopped answering: look for it again
      snprintf(status, sizeof(status), "sensor stopped answering: looking again");
      ok = false;
    }
    return;
  }
  failures = 0;
  Bridge.notify("imu", counter++,
                raw_to_g(be16(&b[0])), raw_to_g(be16(&b[2])), raw_to_g(be16(&b[4])),
                raw_to_dps(be16(&b[8])), raw_to_dps(be16(&b[10])), raw_to_dps(be16(&b[12])));
}
