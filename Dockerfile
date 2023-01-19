FROM ubuntu:20.04 as base_container

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Jupyter and ipykernel
RUN pip3 install ipykernel
RUN pip3 install jupyter

# Set the working directory
WORKDIR /app

# Copy the code and data to the container
ADD . /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# RUN python3 -m pip install --upgrade pip setuptools wheel

# Switch back to jovyan to avoid accidental container runs as root
# USER $NB_UID
CMD ["/bin/bash", "jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]