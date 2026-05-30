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
  return InitResult::SUCCESS;
}