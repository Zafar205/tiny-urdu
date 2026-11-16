from datasets import load_dataset_builder
b=load_dataset_builder("kaab4321/UrduTTS")
print(b.info.features); print(b.info.splits)
