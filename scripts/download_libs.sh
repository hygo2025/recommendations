#!/bin/bash

AWS_SDK_VERSION=${AWS_SDK_VERSION:-1.12.699}
HADOOP_VERSION=${HADOOP_VERSION:-3.3.4}
LIB_INSTALL_PATH=${LIB_INSTALL_PATH:-~/.local/spark/lib-jars/}

mkdir -p "$LIB_INSTALL_PATH"

wget "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" -P "$LIB_INSTALL_PATH"
wget "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" -P "$LIB_INSTALL_PATH"
wget "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/${AWS_SDK_VERSION}/aws-java-sdk-${AWS_SDK_VERSION}.jar" -P "$LIB_INSTALL_PATH"
wget "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-core/${AWS_SDK_VERSION}/aws-java-sdk-core-${AWS_SDK_VERSION}.jar" -P "$LIB_INSTALL_PATH"
wget "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-s3/${AWS_SDK_VERSION}/aws-java-sdk-s3-${AWS_SDK_VERSION}.jar" -P "$LIB_INSTALL_PATH"

chown $(whoami):$(whoami) -R "$LIB_INSTALL_PATH"