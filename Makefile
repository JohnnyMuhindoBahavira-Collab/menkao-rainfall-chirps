.PHONY: install run-all clean

install:
	python -m pip install -r requirements.txt

run-all:
	python scripts/run_all.py

clean:
	rm -rf outputs/*
	touch outputs/.gitkeep
