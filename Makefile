# Compiler and flags
CXX = g++
CXXFLAGS = -Iinclude -I/usr/include/eigen3 -std=c++17 -Wall -Wextra

# Directories
SRC_DIR = src
TEST_DIR = tests
INCLUDE_DIR = include
BUILD_DIR = build
LOG_DIR = logs

# Target executable for tests
TARGETS = test_gpt

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
SRC_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(TEST_FILES))

# Default target
all: $(TARGETS)

# Rule to build each test executable
test_%: $(BUILD_DIR)/test_%.o $(SRC_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule to compile test object files
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/test_%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile source object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build artifacts and logs
clean:
	rm -rf $(BUILD_DIR) $(TARGETS)
	rm -rf $(LOG_DIR)/*.log

