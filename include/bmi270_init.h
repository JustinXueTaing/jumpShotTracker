#pragma once
#include <Arduino.h>
#include "bmi270.h"

enum class InitResult {
    SUCCESS,
    SENSOR_NOT_DETECTED,
    REGISTER_VERIFY_FAIL,
    FIFO_FAIL,
};

InitResult initBMI270(bmi2_dev& sensor);