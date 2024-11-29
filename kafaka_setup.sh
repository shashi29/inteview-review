#!/bin/bash

# Kafka Setup Script

# Configuration
KAFKA_VERSION="3.6.1"
SCALA_VERSION="2.13"
DOWNLOAD_URL="https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
INSTALL_DIR="/opt/kafka"

# Check Prerequisites
check_prerequisites() {
    command -v java >/dev/null 2>&1 || { echo "Java is required. Install Java 8+."; exit 1; }
    command -v wget >/dev/null 2>&1 || { echo "Wget is required. Please install."; exit 1; }
}

# Download and Install Kafka
install_kafka() {
    mkdir -p ${INSTALL_DIR}
    wget ${DOWNLOAD_URL} -O /tmp/kafka.tgz
    tar -xzf /tmp/kafka.tgz -C ${INSTALL_DIR} --strip-components=1
    rm /tmp/kafka.tgz
}

# Configure Kafka
configure_kafka() {
    cat > ${INSTALL_DIR}/config/server.properties << EOL
broker.id=0
listeners=PLAINTEXT://localhost:9092
log.dirs=${INSTALL_DIR}/logs
zookeeper.connect=localhost:2181
num.partitions=1
EOL
    mkdir -p ${INSTALL_DIR}/logs
}

# Main Setup Process
main() {
    check_prerequisites
    install_kafka
    configure_kafka
    echo "Kafka ${KAFKA_VERSION} installed successfully in ${INSTALL_DIR}"
}

# Execute
main