# AgriSegmentation

## Introduction
[ACRE](https://competitions.codalab.org/competitions/27176) is the Agri-food Competition for Robot Evaluation, part of the METRICS project funded by the <b>European Union’s Horizon 2020 research and innovation program under grant agreement No 871252</b>. Autonomous robots compete to demonstrate their ability to perform agricultural tasks (such as removing weeds or surveying crops down to individual-plant resolution). At field campaigns, participants collect data that are then made available like the one you are seeing. For more information about ACRE and METRICS visit the [official website](https://metricsproject.eu/agri-food/).

After years of decline, the number of undernourished people began to slowly increase again in 2015. Food Security requires that everyone can have enough food produced in a sustainable manner. The topic is increasingly gaining attention as food scarcity is worsened by a continuously growing population. Also, food production is threatened by climate change. The topic is so relevant that is part of one of the 17 Sustainable Development Goals of the UN 2030 Agenda. In particular, Food Security is a pillar of SDG number 2, Zero Hunger.

In this context, the agricultural sector is going under a process of revolution by the introduction of digital technologies. The Digital Agricultural Revolution can help to reduce the use of resources (water, fertilizers, and pesticides), thus diminishing the environmental contamination and the costs for the farmers. Also, it could increase the climate resilience of crops and their productivity.

Automatic crop and weed segmentation can be a driver of innovations to optimize the agricultural processes.

## Dataset
The data set we chose to focus on is the Bipbip Maize data set. It is constituted of 90 images and the corresponding
90 target masks. Unfortunately the classes to be segmented are highly unbalanced due to the most frequent presence
of the Background class.

Input image             |  Target segmented mask
:-------------------------:|:-------------------------:
<img src="/Dataset/Training/Bipbip/Mais/Images/Bipbip_mais_im_01391.jpg" width="256" height="192">  |  <img src="/Dataset/Training/Bipbip/Mais/Masks/Bipbip_mais_im_01391.jpg" width="256" height="192">

The data set was augmented, as shown in the following figures, through default pre-processing functions dependent
on the model to be used and they were:
- tensorow.keras.applications.densenet.preprocess input
- tensorow.keras.applications.resnet.preprocess input

While the RGB images were of an original dimension of about 2048x1536, we choose to handle 1024x768 RGB
images so that we are able to loose the minimum of information from the original, double-sized, input images.

## Facing unbalance
To solve the highly unbalance of the classes to be segmented we began at first to sample the entire data set by
simply counting the corresponding encounters of classes pixel by pixel and writing this count in a dictionary
composed of three keys, one per class: 0,1 and 2. After having done so we exploit this precise sampling to
compute the weights to assign to each class as function of their frequency.
Then, to be more precise a re-weighting for each single sample has been computed.

Now that we had the weight to fix the unbalance for each single sample in training, a new loss function to
handle this was needed. In Tensorow/Keras there is no existing loss to handle such specific type of weighting and the existing
parameters class weight and sample weight, in the Tensorow/Keras' method Model.fit, do not work at all for our
specific dynamic case and type of data (lots of issues present at their GitHub repositories). So, we wrote our new
loss function which simply added the notion of weight to the output of any kind of existing loss function passed
as argument.

The full code of this custom weighted Sparse Categorical Cross Entropy loss and the re-weighting process is shown here below, as well as in the uploaded notebooks.

```python
def compute_segmentation_masks_count(img_path):
        mask_img = Image.open(img_path)
        mask_img = mask_img.resize([img_h, img_w], resample=Image.NEAREST)
        mask_arr = np.array(mask_img)
        new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)
        new_mask_arr[np.where(np.all(mask_arr == [216, 124, 18], axis=-1))] = 0
        new_mask_arr[np.where(np.all(mask_arr == [255, 255, 255], axis=-1))] = 1
        new_mask_arr[np.where(np.all(mask_arr == [216, 67, 82], axis=-1))] = 2
        unique, counts = np.unique(new_mask_arr, return_counts=True)
        while len(counts) < 3:
            counts = np.append(counts, [1]) #Laplace smoothing like
        return counts

def compute_weight_unbalance_matrix(filenames):
        counts = []
        for img_path in filenames:
            count = compute_segmentation_masks_count(img_path
                    .replace("jpg", "png").replace("Images", "Masks"))
            count_norm = count / np.linalg.norm(count)
            count_norm[1] = count_norm[0] / count_norm[1]
            count_norm[2] = count_norm[0] / count_norm[2]
            counts.append(count_norm)
        return np.asarray(counts)

def weightedLoss(originalLossFunc):

    def lossFunc(true, pred):

        classSelectors = true #being sparse categorical cross entropy, no argmax here
        weightsList = next(sample_weights_iterator)
        
        classSelectors = [K.equal(tf.cast(i, tf.float32), classSelectors) 
                          for i in range(len(weightsList))]

        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]
        
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
        
    return lossFunc
```

## Model Choice
The first architecture used was a VGG16 with 5 layers to to do up sampling and perform segmentation on the
produced activation masks. We have been using 3x3 filter. We used the fine-tuning technique to better refine also
the weights of the base model other than the ones used to perform the segmentation.
We then moved to try difierent kinds of models, the one we focused more out attention were ResNet152V2 and
DenseNet201 because they gave us the best overall results.
For each various model we always applied fine-tuning and we always added some regularization factor such as L2
regularization and Dropout layers in order to reduce over fitting.

These two final models are composed by a top model of 5 decoders (UpSampling and Conv2D) and LeakyReLU
as activation function which is known to perform slightly better than ReLU. Batch-Normalization was already
present in the base models, which were fine-tuned, to improve the regularization, particularly the robustness to
co-variate shift.
Moreover, on top of these layers we added Dropout with 0.2 rates to further regularize the model and prevent over
fitting together with the L2 regularization applied also to the base model. These values showed to perform better.

## Initialization
## Initialization
For the weight initialization with Xavier Initialization we used GlorotNormal to better initialize weights W and
letting backpropagation algorithm start in advantage position, given that final result of gradient descent is affected
by weights initialization.<br>
![equation](https://latex.codecogs.com/gif.latex?W%20%5Csim%20%5Cmathcal%7BN%7D%5Cleft%28%5Cmu%3D0%2C%5C%2C%20%5C%3B%5Csigma%5E%7B2%7D%3D%5Cfrac%7B2%7D%7BN_%7Bin%7D%20&plus;%20N_%7Bout%7D%7D%5Cright%29)

## Results
With the first architecture using DenseNet201 we were able to reach on CodaLab an IoU on the test data set of
77.57%, while with the second architecture using as base model ResNet152V2, we've reached an IoU on the test
set of 77.61%.
They are indeed very similar models, also in the metrics acquired and shown in the Figures 7 and 8 below. One
could say that at the cost of slightly worse results it's preferable the model with DenseNet201 as base due to its
smaller number of parameters with respect to ResNet152V2 which means reduce prediction time and potential
reduce of further small over fitting.
![densenet](/results/densenet.png)
![resnet](/results/resnet.png)

Given their really high similarity we tried also to apply a Stacked Ensembling model, using a meta learner to
be trained in order to have a model assigning the best possible weights to these two models at prediction time.
Unfortunately we only improved the global result on the overall data set to 0.0634313027 but not on the one we
chose to specifically train onto.

## Possible improvements
We want the network to see most of the possible details, so one possible improvement would be to use the tiling
technique so that we would be able to virtually train the network on the full-sized images and so let it being able
to spot most of details. Although, this require more work to be consistent with the custom weighted loss during
the training process.
