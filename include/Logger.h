#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <chrono>
#include <iomanip>

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

    Logger(const std::string& file_path, LogLevel log_level = LogLevel::INFO);
    ~Logger();

    std::string level_to_string(LogLevel level) const;
    std::string current_timestamp() const;

public:
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static Logger& get_instance(const std::string& file_path = "logs/project.log",
                                LogLevel log_level = LogLevel::DEBUG);

    void log(const std::string& message, LogLevel message_level = LogLevel::INFO);

};

#endif
