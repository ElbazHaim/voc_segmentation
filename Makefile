install:
	pip install --upgrade pip
	pip install pytest black pylint
	pip install -r requirements.txt

test:
	python3 -m pytest -vv test_*.py

format:
	@find "$(PWD)" -name "*.py" -exec black --line-length=79 {} +

lint:
	pylint --output-format=colorized --disable=R,C *.py

traindev:
	python train.py --dev
	
train:
	python train.py	

tune:
	python tune.py

clean:
	rm -r data &&\
	rm -r lightning_logs &&\
	rm -r tb_logs &&\
	rm -r __pycache__

download:
	python datamodules/voc_dataset.py

tensorboard:
	tensorboard --logdir="/home/haim/code/voc_segmentation/lightning_logs/" --host localhost --port 8080