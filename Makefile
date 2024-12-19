build:
	gcc -Wall -Wextra -pedantic -shared -o image_operations.so -fPIC image_operations.c 

run:
	python gui.py

clean:
	rm -f image_operations.so
