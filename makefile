install:
	pip install --upgrade pip &&\
		pip install pytest
		pip install black
		pip install pylint
		pip install -r requirements.txt

test:
	python3 -m pytest -vv test_*.py

format:
	@find "$(PWD)" -name "*.py" -exec black {} +

lint:
	pylint --output-format=colorized --disable=R,C *.py

devrun:
	python main.py --dev
	
run:
	python main.py	
	
eda:
	python eda.py

clean:
	rm -r data &&\
	rm -r lightning_logs &&\
	rm -r tb_logs &&\
	rm -r __pycache__

download:
	python datamodules/voc_dataset.py
