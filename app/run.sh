#!/bin/bash

# Script to run app.py and rabbitmq_interview_review_service.py concurrently

# Define log files for each process
APP_LOG="app.log"
RABBITMQ_LOG="rabbitmq_service.log"

# Function to handle cleanup when script is terminated
cleanup() {
    echo "Shutting down services..."
    kill $APP_PID $RABBITMQ_PID 2>/dev/null
    exit 0
}

# Set up trap to catch SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Clear previous log files if they exist
> $APP_LOG
> $RABBITMQ_LOG

# Start app.py and save its PID
echo "Starting app.py..."
uvicorn app:app --host 0.0.0.0 --port 8082 > $APP_LOG 2>&1 &
APP_PID=$!

# Start rabbitmq_interview_review_service.py and save its PID
echo "Starting rabbitmq_interview_review_service.py..."
python rabbitmq_interview_review_service.py > $RABBITMQ_LOG 2>&1 &
RABBITMQ_PID=$!

# Check if both processes started successfully
if ! ps -p $APP_PID > /dev/null; then
    echo "Failed to start app.py"
    cleanup
fi

if ! ps -p $RABBITMQ_PID > /dev/null; then
    echo "Failed to start rabbitmq_interview_review_service.py"
    cleanup
fi

echo "All services running. App PID: $APP_PID, RabbitMQ Service PID: $RABBITMQ_PID"
echo "Press Ctrl+C to stop all services"

# Wait for both processes to complete (or until script is terminated)
wait $APP_PID $RABBITMQ_PID

echo "All services have completed execution"