# Compiler and flags
CXX = g++
CXXFLAGS = -Iinclude -I/usr/include/eigen3 -std=c++17 -Wall -Wextra
LDFLAGS = 
BUILD_MODE = DEBUG


ifeq ($(BUILD_MODE), DEBUG)
    CXXFLAGS += -g -O0
else ifeq ($(BUILD_MODE), RELEASE)
    CXXFLAGS += -O3
endif

# Directories
SRC_DIR = src
TEST_DIR = tests
INCLUDE_DIR = include
BUILD_DIR = build
LOG_DIR = logs

# Source and test files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
SRC_OBJECTS = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_FILES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Executables (auto-detect from test files)
TEST_TARGETS = $(TEST_FILES:$(TEST_DIR)/%.cpp=%)

# Default target
all: $(TEST_TARGETS)

# Rule for building test executables
%: $(BUILD_DIR)/%.o $(SRC_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule for compiling source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(LOG_DIR)

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TEST_TARGETS)
	rm -rf $(LOG_DIR)/*.log

# Phony targets
.PHONY: all clean
