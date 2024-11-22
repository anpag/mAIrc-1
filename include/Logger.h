#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <string>
#include <fstream>

enum class LogLevel {
    INFO,
    DEBUG,
    WARNING,
    ERROR
};

class Logger {
private:
    std::ofstream log_file;
    LogLevel level;

    std::string level_to_string(LogLevel level) const;

public:
    Logger(const std::string& file_path, LogLevel log_level = LogLevel::INFO);
    ~Logger();

    void log(const std::string& message, LogLevel message_level = LogLevel::INFO);
};

#endif
