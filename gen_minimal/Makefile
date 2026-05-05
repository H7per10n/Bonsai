CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99

# Build all models
all: xgboost_binary/test_model lightgbm_binary/test_model xgboost_multiclass/test_model lightgbm_multiclass/test_model xgboost_regression/test_model lightgbm_regression/test_model

# Individual model targets
xgboost_binary/test_model: xgboost_binary/main.c xgboost_binary/xgboost.h
	$(CC) $(CFLAGS) -o xgboost_binary/test_model xgboost_binary/main.c -lm

lightgbm_binary/test_model: lightgbm_binary/main.c lightgbm_binary/lightgbm.h
	$(CC) $(CFLAGS) -o lightgbm_binary/test_model lightgbm_binary/main.c -lm

xgboost_multiclass/test_model: xgboost_multiclass/main.c xgboost_multiclass/xgboost.h
	$(CC) $(CFLAGS) -o xgboost_multiclass/test_model xgboost_multiclass/main.c -lm

lightgbm_multiclass/test_model: lightgbm_multiclass/main.c lightgbm_multiclass/lightgbm.h
	$(CC) $(CFLAGS) -o lightgbm_multiclass/test_model lightgbm_multiclass/main.c -lm

xgboost_regression/test_model: xgboost_regression/main.c xgboost_regression/xgboost.h
	$(CC) $(CFLAGS) -o xgboost_regression/test_model xgboost_regression/main.c -lm

lightgbm_regression/test_model: lightgbm_regression/main.c lightgbm_regression/lightgbm.h
	$(CC) $(CFLAGS) -o lightgbm_regression/test_model lightgbm_regression/main.c -lm

# Clean targets
clean: clean-xgboost_binary clean-lightgbm_binary clean-xgboost_multiclass clean-lightgbm_multiclass clean-xgboost_regression clean-lightgbm_regression

clean-xgboost_binary:
	rm -f xgboost_binary/test_model

clean-lightgbm_binary:
	rm -f lightgbm_binary/test_model

clean-xgboost_multiclass:
	rm -f xgboost_multiclass/test_model

clean-lightgbm_multiclass:
	rm -f lightgbm_multiclass/test_model

clean-xgboost_regression:
	rm -f xgboost_regression/test_model

clean-lightgbm_regression:
	rm -f lightgbm_regression/test_model

# Run targets
run-all: run-xgboost_binary run-lightgbm_binary run-xgboost_multiclass run-lightgbm_multiclass run-xgboost_regression run-lightgbm_regression

run-xgboost_binary: xgboost_binary/test_model
	@echo "Running xgboost_binary:"
	@./xgboost_binary/test_model
	@echo

run-lightgbm_binary: lightgbm_binary/test_model
	@echo "Running lightgbm_binary:"
	@./lightgbm_binary/test_model
	@echo

run-xgboost_multiclass: xgboost_multiclass/test_model
	@echo "Running xgboost_multiclass:"
	@./xgboost_multiclass/test_model
	@echo

run-lightgbm_multiclass: lightgbm_multiclass/test_model
	@echo "Running lightgbm_multiclass:"
	@./lightgbm_multiclass/test_model
	@echo

run-xgboost_regression: xgboost_regression/test_model
	@echo "Running xgboost_regression:"
	@./xgboost_regression/test_model
	@echo

run-lightgbm_regression: lightgbm_regression/test_model
	@echo "Running lightgbm_regression:"
	@./lightgbm_regression/test_model
	@echo

.PHONY: all clean clean-xgboost_binary clean-lightgbm_binary clean-xgboost_multiclass clean-lightgbm_multiclass clean-xgboost_regression clean-lightgbm_regression run-all run-xgboost_binary run-lightgbm_binary run-xgboost_multiclass run-lightgbm_multiclass run-xgboost_regression run-lightgbm_regression
