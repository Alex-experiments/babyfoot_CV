install:
	cd dataset_creation && git clone https://github.com/2vin/yolo_annotation_tool.git

create_dataset:
	@echo "Creation of the dataset_creation"
	python3 dataset_creation/frames_extraction.py
