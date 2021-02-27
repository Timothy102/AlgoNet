build:
	rm -rf logs
	sudo groupadd docker
	docker build -t sorting .
	docker run --rm tim .
