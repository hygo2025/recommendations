FROM spark:3.5.4-python3

ENV AWS_SDK_VERSION=1.12.699
ENV HADOOP_VERSION=3.3.4
ENV LIB_INSTALL_PATH=/opt/spark/lib-jars/

USER root

RUN apt-get update && \
    apt-get install -y build-essential python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \
    pip install --upgrade pip

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

RUN chmod +x /app/scripts/download_libs.sh
RUN /app/scripts/download_libs.sh

# Variáveis de ambiente
ENV PYTHONPATH=/app

RUN chown spark:spark -R ${LIB_INSTALL_PATH}

USER spark
