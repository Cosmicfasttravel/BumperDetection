#include <string>
#include <unordered_map>
#ifndef WIN32
#include "networktables/NetworkTableInstance.h"
#include <networktables/DoubleTopic.h>
#include <networktables/StringTopic.h>
#include "config_extraction.h"

void setupNT(){
auto nt = nt::NetworkTableInstance::GetDefault();
    nt.StartClient4("bumper_detection");
    nt.SetServer("det", 0);
}

void publishRobotPos(const std::string& label, double x, double y, double z) {
    static auto nt = nt::NetworkTableInstance::GetDefault();
    auto table = nt.GetTable("bumper_detection/" + label);

    static std::unordered_map<std::string, nt::DoublePublisher> xPubs, yPubs, zPubs;

    if (!xPubs.count(label)) {
        xPubs[label] = table->GetDoubleTopic("x").Publish();
        yPubs[label] = table->GetDoubleTopic("y").Publish();
        zPubs[label] = table->GetDoubleTopic("z").Publish();
    }

    xPubs[label].Set(x);
    yPubs[label].Set(y);
    zPubs[label].Set(z);
}

#endif