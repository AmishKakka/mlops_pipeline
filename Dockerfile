FROM ubuntu:22.04

ARG TARGETPLATFORM=linux/amd64,linux/arm64
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget gnupg software-properties-common && \
    apt-get install -y python3-pip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    echo "Python installed" && \ 
    python3 --version

COPY requirements.txt .
RUN echo "Installing libraries..." && \
    pip install -r requirements.txt && \
    echo "Installed libraries..."

WORKDIR /mlops_pipeline
COPY . .

CMD ["tail", "-f", "/dev/null"]