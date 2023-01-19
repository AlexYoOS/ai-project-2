# python-docker-jupyter

Used the docker file from jupyter's implementation of docker templates https://github.com/jupyter/docker-stacks/tree/master/minimal-notebook

~~docker build --rm -f "Dockerfile" -t pythondockerjupyter:latest "."~~

~~docker run -p 8888:8888 pythondockerjupyter:latest~~

To start notebook run

`auto/dev` (builds docker image first time and start juptyer notebook)

To rebuild docker image and start the notebook, run

`auto/dev rebuild`

Open in any browser to access Jupyter notebook
http://localhost:10000/lab
