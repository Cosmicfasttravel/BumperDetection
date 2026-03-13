#ifndef BUMPERDETECTION_CONFIG_EXTRACTION_H
#define BUMPERDETECTION_CONFIG_EXTRACTION_H
#include <filesystem>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

std::string extractByTag(const std::string& tag);
void pollForChanges();

#endif //BUMPERDETECTION_CONFIG_EXTRACTION_H