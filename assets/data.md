# Data Preparation

We put all data under the `datasets` folder. The complete data structure looks like this.
```
${SimOWT_ROOT}
    -- datasets
        -- bdd
        -- coco
            -- annotations
            -- train2017
            -- val2017
        -- tao
            -- annotations
            -- frames
```


## Pretrained weights
The pretrained backbone weight can be downloaded by the following link.
```
https://drive.google.com/file/d/10vECiZFWAqVP8plwpZ9t8MczcFHCLmlN/view?usp=drive_link
```

### COCO
Please download [coco](https://cocodataset.org/#home) from the offical website. We expect that the data is organized as below.
```
${SimOWT_ROOT}
    -- datasets
        -- coco
            -- annotations
            -- train2017
            -- val2017
```

### TAO
Please download [TAO](http://taodataset.org/) from the offical website.  We expect that the data is organized as below.
```
${SimOWT_ROOT}
    -- datasets
        -- tao
            -- annotations
            -- frames
```
### Further -- datasets in class-agnostic
Since the OWT task is performed in the form of class-agnostic, we provide the corresponding class-agnostic json file to facilitate your training and verification. The json files can be downloaded by the following link.
```
https://drive.google.com/file/d/1h2rohSDvJv_XvUdTaux5rN9jx86rzR9g/view?usp=sharing
```

### TrackEval validation.json
The validation file under {TrackEval/data/pse/idol/score} can be downloaded by the following link.
```
https://drive.google.com/file/d/1aNW9X-CUMQqVZhHKKz55VLXQVEY0ZEQb/view?usp=sharing
```
