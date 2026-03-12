#ifndef BUMPERDETECTION_SEND_DATA_H
#define BUMPERDETECTION_SEND_DATA_H
#include <string>
void setupNT();
void publishRobotPos(const std::string& label, double x, double y, double z);
#endif
