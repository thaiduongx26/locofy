sudo docker stop sequence_labeling_service
sudo docker rm sequence_labeling_service
sudo docker build -t sequence_labeling_service -f Dockerfile .
sudo docker run --restart unless-stopped --network host --name sequence_labeling_service -p 8003:8003 sequence_labeling_service
