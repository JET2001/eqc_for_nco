FROM python:3.8-slim
WORKDIR /home

RUN apt upgrade -y \
    && apt update \
    && apt install git-all -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install libfreetype6-dev libpng-dev -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements2.txt ./
RUN pip install --no-cache-dir --default-timeout=100 -r requirements2.txt
CMD ["/bin", "bash"]