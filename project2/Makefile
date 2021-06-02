.PHONY: project.pdf clean

default: all

all: project.pdf

project.pdf: project.ipynb
	jupyter nbconvert --to pdf $< \
      --TagRemovePreprocessor.remove_cell_tags remove_cell \
      --TagRemovePreprocessor.remove_input_tags remove_input

clean:
	rm project.pdf
