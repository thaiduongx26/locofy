# How to run it?
```
# python 3.9
pip install -r requirements.txt
```
### Training:
- If you don't want to use my data. Make sure you have `Data` folder
```
# Generate kfolds data
python data_split.py
```
- If you wanna use my data, please skip the step above.
```
# Please ping me for the permission key.
dvc pull
```
- Training script:
```
mkdir Output
bash train.sh
```
### Running app for serving models
```
bash run.sh
```
- It gonna run in port 8003. I'm also running it following URL: `http://34.121.223.86:8003/inference`. Example:
```
curl --header "Content-Type: application/json" --request POST --data '{"input": [["a", 58], ["0", 12], ["b",7], ["0",32], ["c",292]], "direction": "horizontal"}' http://34.121.223.86:8003/inference
```
- Response: `{"output":[["a","b"],"c"]}`

# Approach detail
### Output definition
Because in each sample, it's always following one direction, we just have the object's length and the size of space between them. So I formulate this problem into a sequence labeling problem, the purpose is that we just need to classify each object is the start of a group or not. Then we have 2 classes: `Begin Group (B-Group)` and `Inside Group (I-Group)`. If we have 2 consecutive `B-Group`, that means the first one is a group containing only 1 object. Example:
```
list object = ["a", "b", "c", "d", "e", "f"]
prediction = [B-Group, I-Group, I-Group, B-Group, B-Group, I-Group]
=> [["a", "b", "c"], "d", ["e", "f"]]
```

### Features selection
Because each sample only has 1 direction, we don't have the height in the case of horizontal and width in the case vertical. So I focus on 3 features that can measure the shape and distance between them: object size, object's left space, and direction. I chose the max value of each feature based on the current distribution and we have an edge case in the private test, it should be rescaled and normalized.

### Model architecture
Firstly, for each element in the sequence, I create an embedding layer for each feature and get the sum of them, so after that, we have an embedding present for an element. For the model, I use a small transformer encoder model combined with Sequence Labeling Head. For the metric, I used entity exact match level to calculate the accuracy and f1-score, it means a group was wrong if that group contain more or less objects, even only 1.

![alt text]([http://url/to/img.png](https://github.com/thaiduongx26/locofy/blob/main/doc_images/locofy_model%20(1).png)https://github.com/thaiduongx26/locofy/blob/main/doc_images/locofy_model%20(1).png)
