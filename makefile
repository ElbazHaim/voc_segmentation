install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python3 -m pytest -vv test_*.py

format:
	black *.py

lint:
	pip install pylint &&\
		pylint --output-format=colorized --disable=R,C *.py

devrun:
	python main.py --dev
	
run:
	python main.py	

clean:
	rm -r data &&\
	rm -r lightning_logs &&\
	rm -r tb_logs &&\
	rm -r __pycache__

all:
	make install test format lint devrun
