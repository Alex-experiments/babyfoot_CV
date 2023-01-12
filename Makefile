install:
	pip install poetry
	poetry install || true
	git clone https://github.com/GuillaumeDugat/baby-foot-dataset.git

extract_frames:
	poetry run python3 dataset_creation/frames_extraction.py

annotate_images:
	poetry run python3 dataset_creation/images_annotation.py

pull-dataset:
	cd baby-foot-dataset && git pull origin main

push-dataset:
	cd baby-foot-dataset && git add * && git commit -m "Add images" && git push origin main
