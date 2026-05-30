#pragma once
#include <Arduino.h>

extern "C" {
    #include "bmi270.h"
    #include "bmi2.h"
}

BMI2_INTF_RETURN_TYPE bmi270_i2c_read(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr);

BMI2_INTF_RETURN_TYPE bmi270_i2c_write(uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr);

void bmi270_delay_us(uint32_t period, void *intf_ptr);