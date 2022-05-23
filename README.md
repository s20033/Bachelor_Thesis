# Bachelor_Thesis
A study on effects of Data Augmentation in Classification Accuracy in Deep Learning.

# Introduction
The data augmentation is an important part of deep learning and computer vision.
The techniques of data augmentation are also a widely used method in the most
popular deep learning models. However there are very little studies done on the
effects of data augmentation techniques have on the accuracy of the model. This
thesis is mostly based on investigational study on data augmentation in classification
accuracy in deep learning. The objective of this thesis is to investigate the different
data augmentation techniques and their effects in learning process of VGGNet model,
performance generalization, prediction and other behaviour of deep neural networks.

The experiment carried out during the work of this thesis shows that data augmenta-
tion has a very significant impact on the classification accuracy of the VGGNet model.

I also got to know from the experiment in CIFAR10 data, that data augmentation
can improve performance increment with correct tuning and selecting the right kind
of data augmentation algorithm. The dependency on the performance of the model
to give the expected betterment of accuracy is also model-dependent. From the sets
of experiments performed, the general behavior of data augmentation when applied
all in a model, randomly parameterized, does not seem to be a good choice. However,
single data augmentation like random cropping, or random horizontal flipping seems
to be an effective way of enhancing model accuracy for the classification problem.
The CIFAR10 dataset that was used in the experiment is real-world images, which
in the general case, has a common characteristic that the object is mostly centered
in the image. In such a scenario, the random crop data augmentation stands as
the significant technique, because we performed random cropping based on centric
datapoints, this removed the non-significant features or pixels from images and had
significant datapoints left after cropping, which was important for model to extract
important features while learning. To sum up, the results from the experiments
show that data augmentation is beneficial for the result of the learning process if
the data augmentation is wisely chosen and fine-tuned in the right manner. With
no matter of surprise data augmentation techniques are directly proportional to the
characteristic of the dataset, and also the model complexity. From the experiment,
I came to the conclusion that data augmentation is significant in improving the
model accuracy, but not necessarily all data augmentation. This thesis work is
also concluded leaving the spaces of further work and research, there is yet a lot of
research spaces, like parameter tuning in particular data augmentation, analyzing
the data augmentation according to the characteristics of the dataset, and selection
of preprocessing techniques in detail should be carried out, in order to improve the
classification accuracy of models.
