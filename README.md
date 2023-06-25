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

<img src="https://github.com/thaiduongx26/locofy/blob/main/doc_images/locofy_model%20(1).png" width=60% height=60%/>

### Training strategy
- I create 5 folds of data, each fold contains 80% data for training and 20% for testing. Then I got 5 models based on 5 folds and another model trained on full data. In total, we have 6 models.
- For inference, I ensemble all of them by calculating the mean of logit.
<img src="https://github.com/thaiduongx26/locofy/blob/main/doc_images/pipeline.png" width=75% height=75%>

### Some results
|   |Precision|Recall|F1-score|n_groups|
|---|:---:|:---:|:---:|:---:|
| **Fold-0**  |0.9444911690496215|0.9468802698145026|0.9456842105263158|1186|
| **Fold-1**  |0.9649280575539568|0.9503985828166519|0.9576082106202588|1129|
| **Fold-2**  |0.9658119658119658|0.9625212947189097|0.9641638225255974|1174|
| **Fold-3**  |0.9582971329278888|0.9395229982964225|0.9488172043010752|1174|
| **Fold-4**  |0.9743150684931506|0.9619611158072696|0.968098681412165|1183|

# Some ideas
- I don't know why this data only follow 1 direction but I see we have the images, we have the positions, and it's the websites so we can crawl a lot of data. Some important features that we can explore like width, height, the distance between 2 centers of 2 objects, image features (we can see 2 objects relevant or not based on content in image), text features, etc
- Following my approach, there are some ways to improve if I have more time to focus:
  - Applying rules-based for some special cases like objects have size = 0 or set threshold for the ratio of size and space between 2 objects.
  - Investigate more about rescale to choose the better ratio to remove the noise of space between 2 objects.
  - Tunning the params of the model focuses on the num_heads and the num_layers of the encoder, increases the batch size and tuning the dropout layer to avoid overfitting.
 
- Some other approaches ideas:
  - Using detection pre-trained models as the backbone, it has a better measure of the correlation between two objects, both distance and visualization.
  - Combining image embedding and position embedding (or maybe content embedding if we have), it can help to build a graph that maps every object in the webpage. After that we can group the objects.
  - I am not really sure about can we extract the label of each element in the website or not, but I think most websites follow some limited structures (it's large, but I think it's still limited), for example, a group of buttons in the login/register section, something like that. If it is feasible, I think it's possible to pre-train a model that can perform better for this specific downstream task.

## Thank you for reading
