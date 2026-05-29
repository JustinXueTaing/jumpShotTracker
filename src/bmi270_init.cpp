#include <Arduino.h>
#include "bmi270_init.h"    
#include "bmi270_hal.h"
#include "bmi2_defs.h"

enum bmi2_intf {
    BMI2_SPI_INTF = 0,
    BMI2_I2C_INTF,
    BMI2_I3C_INTF
};

enum class InitResult {
    SUCCESS,
    SENSOR_NOT_DETECTED,
    REGISTER_VERIFY_FAIL,
    FIFO_FAIL,
};

struct bmi2_dev dev = {
  .intf = BMI2_I2C_INTF,
  .intf_ptr = &i2c_addr,
  .read = bmi270_i2c_read,
  .write = bmi270_i2c_write,
  .delay_us = bmi270_delay_us
};

InitResult initBMI270(bmi2_dev& sensor) { 
  if (bmi2_soft_reset(&sensor) != BMI2_OK) {
    return InitResult::SENSOR_NOT_DETECTED;
  }

}