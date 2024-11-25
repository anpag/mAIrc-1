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

Logger& Logger::get_instance(const std::string& file_path, LogLevel log_level) {
    static Logger instance(file_path, log_level);
    return instance;
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return  "UNKNOWN";
    }
}

std::string Logger::current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%dT%H:%M:%S") 
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

void Logger::log(const std::string& message, LogLevel message_level) {
    if (message_level >= level) {
        std::string output = "[" + current_timestamp() + "] [" + 
                             level_to_string(message_level) + "] " + message;
        std::cout << output << std::endl;
        if (log_file.is_open()) {
            log_file << output << std::endl;
        }
    }
}

void Logger::set_level(LogLevel log_level) {
    level = log_level;
}