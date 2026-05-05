CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
TARGET = test_model
SOURCES = main.c

all: $(TARGET)

$(TARGET): $(SOURCES) xgboost.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) -lm

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
