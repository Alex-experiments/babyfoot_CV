install:
	pip install poetry
	poetry install || true 

extract_frames:
	poetry run python3 dataset_creation/frames_extraction.py

annotate_images:
	poetry run python3 dataset_creation/images_annotation.py
