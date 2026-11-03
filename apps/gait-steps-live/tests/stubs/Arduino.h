#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
extern uint32_t sim_us;                                   // the simulated clock, in microseconds
inline uint32_t micros() { return sim_us; }
inline void delay(uint32_t ms) { sim_us += ms * 1000; }
inline void delayMicroseconds(uint32_t us) { sim_us += us; }
class String { std::string s; public: String(const char *c = "") : s(c) {} const char *c_str() const { return s.c_str(); } };
