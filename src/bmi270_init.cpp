#include <Arduino.h>
#include "bmi270_init.h"
#include "bmi270_hal.h"

extern "C" {
  #include "bmi270.h"
  #include "bmi2_defs.h"
}


InitResult initBMI270(bmi2_dev& sensor) { 
  if (bmi2_soft_reset(&sensor) != BMI2_OK) {
    return InitResult::SENSOR_NOT_DETECTED;
  }
  if (bmi270_init(&sensor) != BMI2_OK) {
      return InitResult::SENSOR_NOT_DETECTED;
  }
  sensor.delay_us(2000, sensor.intf_ptr);

  // Sensor Enabling
  uint8_t sens_list[2] = {BMI2_ACCEL, BMI2_GYRO};
  if (bmi2_sensor_enable(sens_list, 2, &sensor) != BMI2_OK) {
    return InitResult::SENSOR_NOT_DETECTED;
  }

  //Sensor Activation Testing
  struct bmi2_sens_config configs[2];

  configs[0].type = BMI2_ACCEL;
  configs[0].cfg.acc.odr = BMI2_ACC_ODR_800HZ;

  configs[1].type = BMI2_GYRO;
  configs[1].cfg.gyr.odr = BMI2_GYR_ODR_800HZ;
  configs[1].cfg.gyr.range = BMI2_GYR_RANGE_1000;

  if (bmi270_set_sensor_config(configs, 2, &sensor) != BMI2_OK) {
    return InitResult::REGISTER_VERIFY_FAIL;
  }

  // Sensor Flushing
  if (bmi2_set_command_register(BMI2_FIFO_FLUSH_CMD, &sensor) != BMI2_OK) {
    return InitResult::FIFO_FAIL;
  }
  //Enable accel + Gyro
  uint16_t config_flags = BMI2_FIFO_ACC_EN | BMI2_FIFO_GYR_EN;
  if (bmi2_set_fifo_config(config_flags, BMI2_ENABLE, &sensor) != BMI2_OK) {
    return InitResult::FIFO_FAIL;
  }

  // Set Watermark at 50%
  if (bmi2_set_fifo_wm(3000, &sensor) != BMI2_OK) {
    return InitResult::FIFO_FAIL;
  }
  return InitResult::SUCCESS;
}