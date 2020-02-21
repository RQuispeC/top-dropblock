Torchreid
===========
Torchreid is a library built on `PyTorch <https://pytorch.org/>`_ for deep-learning person re-identification.

It features:

- multi-GPU training
- support both image- and video-reid
- end-to-end training and evaluation
- incredibly easy preparation of reid datasets
- multi-dataset training
- cross-dataset evaluation
- standard protocol used by most research papers
- highly extensible (easy to add models, datasets, training methods, etc.)
- implementations of state-of-the-art deep reid models
- access to pretrained reid models
- advanced training techniques
- visualization tools (tensorboard, ranks, etc.)


Code: https://github.com/KaiyangZhou/deep-person-reid.

Documentation: https://kaiyangzhou.github.io/deep-person-reid/.

How-to instructions: https://kaiyangzhou.github.io/deep-person-reid/user_guide.

Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.


Installation
---------------

Make sure your `conda <https://www.anaconda.com/distribution/>`_ is installed.


.. code-block:: bash

    # cd to your preferred directory and clone this repo
    git clone https://github.com/KaiyangZhou/deep-person-reid.git

    # create environment
    cd deep-person-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop


Get started: 30 seconds to Torchreid
-------------------------------------
1. Import ``torchreid``

.. code-block:: python
    
    import torchreid

2. Load data manager

.. code-block:: python
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )

3 Build model, optimizer and lr_scheduler

.. code-block:: python
    
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

4. Build engine

.. code-block:: python
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

5. Run training and test

.. code-block:: python
    
    engine.run(
        save_dir='log/resnet50',
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


A unified interface
-----------------------
In "deep-person-reid/scripts/", we provide a unified interface to train and test a model. See "scripts/main.py" and "scripts/default_config.py" for more details. "configs/" contains some predefined configs which you can use as a starting point.

Below we provide examples to train and test `OSNet (Zhou et al. ICCV'19) <https://arxiv.org/abs/1905.00953>`_. Assume :code:`PATH_TO_DATA` is the directory containing reid datasets.

Conventional setting
^^^^^^^^^^^^^^^^^^^^^

To train OSNet on Market1501, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA \
    --gpu-devices 0


The config file sets Market1501 as the default dataset. If you wanna use DukeMTMC-reID, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    -s dukemtmcreid \
    -t dukemtmcreid \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA \
    --gpu-devices 0 \
    data.save_dir log/osnet_x1_0_dukemtmcreid_softmax_cosinelr

The code will automatically (download and) load the ImageNet pretrained weights. After the training is done, the model will be saved as "log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250". Under the same folder, you can find the `tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ file. To visualize the learning curves using tensorboard, you can run :code:`tensorboard --logdir=log/osnet_x1_0_market1501_softmax_cosinelr` in the terminal and visit :code:`http://localhost:6006/` in your web browser.

Evaluation is automatically performed at the end of training. To run the test again using the trained model, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --root $PATH_TO_DATA \
    --gpu-devices 0 \
    model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
    test.evaluate True


Cross-domain setting
^^^^^^^^^^^^^^^^^^^^^

Suppose you wanna train OSNet on DukeMTMC-reID and test its performance on Market1501, you can do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
    -s dukemtmcreid \
    -t market1501 \
    --transforms random_flip color_jitter \
    --root $PATH_TO_DATA \
    --gpu-devices 0

Here we only test the cross-domain performance. However, if you also want to test the performance on the source dataset, i.e. DukeMTMC-reID, you can set :code:`-t dukemtmcreid market1501`, which will evaluate the model on the two datasets separately.

Different from the same-domain setting, here we replace :code:`random_erase` with :code:`color_jitter`. This can improve the generalization performance on the unseen target dataset.

Pretrained models are available in the `Model Zoo <https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html>`_.


Datasets
--------

Image-reid datasets
^^^^^^^^^^^^^^^^^^^^^
- `Market1501 <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`_
- `CUHK03 <https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`_
- `DukeMTMC-reID <https://arxiv.org/abs/1701.07717>`_
- `MSMT17 <https://arxiv.org/abs/1711.08565>`_
- `VIPeR <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.7285&rep=rep1&type=pdf>`_
- `GRID <http://www.eecs.qmul.ac.uk/~txiang/publications/LoyXiangGong_cvpr_2009.pdf>`_
- `CUHK01 <http://www.ee.cuhk.edu.hk/~xgwang/papers/liZWaccv12.pdf>`_
- `SenseReID <http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf>`_
- `QMUL-iLIDS <http://www.eecs.qmul.ac.uk/~sgg/papers/ZhengGongXiang_BMVC09.pdf>`_
- `PRID <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_

Video-reid datasets
^^^^^^^^^^^^^^^^^^^^^^^
- `MARS <http://www.liangzheng.org/1320.pdf>`_
- `iLIDS-VID <https://www.eecs.qmul.ac.uk/~sgg/papers/WangEtAl_ECCV14.pdf>`_
- `PRID2011 <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_
- `DukeMTMC-VideoReID <http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf>`_

Models
-------

ImageNet classification models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `ResNet <https://arxiv.org/abs/1512.03385>`_
- `ResNeXt <https://arxiv.org/abs/1611.05431>`_
- `SENet <https://arxiv.org/abs/1709.01507>`_
- `DenseNet <https://arxiv.org/abs/1608.06993>`_
- `Inception-ResNet-V2 <https://arxiv.org/abs/1602.07261>`_
- `Inception-V4 <https://arxiv.org/abs/1602.07261>`_
- `Xception <https://arxiv.org/abs/1610.02357>`_

Lightweight models
^^^^^^^^^^^^^^^^^^^
- `NASNet <https://arxiv.org/abs/1707.07012>`_
- `MobileNetV2 <https://arxiv.org/abs/1801.04381>`_
- `ShuffleNet <https://arxiv.org/abs/1707.01083>`_
- `ShuffleNetV2 <https://arxiv.org/abs/1807.11164>`_
- `SqueezeNet <https://arxiv.org/abs/1602.07360>`_

ReID-specific models
^^^^^^^^^^^^^^^^^^^^^^
- `MuDeep <https://arxiv.org/abs/1709.05165>`_
- `ResNet-mid <https://arxiv.org/abs/1711.08106>`_
- `HACNN <https://arxiv.org/abs/1802.08122>`_
- `PCB <https://arxiv.org/abs/1711.09349>`_
- `MLFN <https://arxiv.org/abs/1803.09132>`_
- `OSNet <https://arxiv.org/abs/1905.00953>`_

Losses
------
- `Softmax (cross entropy loss with label smoothing) <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf>`_
- `Triplet (hard example mining triplet loss) <https://arxiv.org/abs/1703.07737>`_


Citation
---------
If you find this code useful to your research, please cite the following publication.

.. code-block:: bash
    
    @article{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      journal={arXiv preprint arXiv:1905.00953},
      year={2019}
    }
