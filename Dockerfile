FROM python:3.13-slim

# Install git, build dependencies, and certificates
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Install the git dependency first
RUN pip install --no-cache-dir git+https://github.com/snkas/exputilpy.git@312749788b8d8d9d38cc4db9991193b84831d919

# Install remaining requirements, excluding the git dependency
RUN pip install --no-cache-dir -r requirements.txt || \
    (grep -v "exputil @" requirements.txt > requirements_no_git.txt && \
     pip install --no-cache-dir -r requirements_no_git.txt)

COPY src/ ./src/
COPY pyproject.toml ./
COPY lrsim_logo.png ./
COPY README.md ./

# Create output directories
RUN mkdir -p /app/logs /app/tle_output /app/visualisation_output

# Environment variables for common configuration
ENV CONFIG_FILE=src/config/ether_simple.yaml
ENV PYTHONPATH=/app

# Set the entrypoint to run the simulator with configurable config file
ENTRYPOINT ["python", "-m", "src.main", "--config"]
CMD ["src/config/ether_simple.yaml"]
