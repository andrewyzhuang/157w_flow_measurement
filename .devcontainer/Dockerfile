FROM mcr.microsoft.com/devcontainers/miniconda:latest

# Copy the environment.yml into the container
COPY environment.yml /tmp/environment.yml

# Create the Conda environment
RUN conda env create -f /tmp/environment.yml

# Prevent base environment from auto-activating
RUN conda config --system --set auto_activate_base false

# Initialize Conda for bash and auto-activate the environment
RUN conda init bash && \
    echo "conda activate flow" >> ~/.bashrc