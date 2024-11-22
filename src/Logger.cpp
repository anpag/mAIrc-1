#include "Logger.h"

Logger::Logger(const std::string& file_path, LogLevel log_level)
    : level(log_level) {
    log_file.open(file_path, std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error: Could not open log file." << std::endl;
        exit(1);
    }
}

Logger::~Logger() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void Logger::log(const std::string& message, LogLevel message_level) {
    if (message_level >= level) {
        std::string output = "[" + level_to_string(message_level) + "] " + message;
        std::cout << output << std::endl;  // Print to console
        if (log_file.is_open()) {
            log_file << output << std::endl;  // Write to file
        }
    }
}
