import sys
import os
import os.path as osp
import warnings
import time
import argparse

import torch
import torch.nn as nn

from default_config import (
    get_default_config, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.model.name.startswith("fgnet"):
            if cfg.loss.name == 'softmax':
                engine = torchreid.engine.ImageSoftmaxFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'softmax_npairs':
                engine = torchreid.engine.ImageSoftmaxNpairsFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    margin_sasc=cfg.loss.npairs.margin_sasc,
                    margin_sadc=cfg.loss.npairs.margin_sadc,
                    margin_dasc=cfg.loss.npairs.margin_dasc
                )
            elif cfg.loss.name == 'triplet_npairs':
                engine = torchreid.engine.ImageTripletNpairsFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_n=cfg.loss.triplet.weight_n,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    margin_sasc=cfg.loss.npairs.margin_sasc,
                    margin_sadc=cfg.loss.npairs.margin_sadc,
                    margin_dasc=cfg.loss.npairs.margin_dasc
                )
            elif cfg.loss.name == 'triplet_softmax':
                engine = torchreid.engine.ImageTripletSoftmaxFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_x_parts=cfg.loss.triplet.weight_x_parts,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'triplet_softmax_dropbatch':
                engine = torchreid.engine.ImageTripletSoftmaxDropbatchFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_x_parts=cfg.loss.triplet.weight_x_parts,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'triplet_softmax_parts_dropbatch':
                engine = torchreid.engine.ImageTripletSoftmaxPartsDropbatchFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_x_parts=cfg.loss.triplet.weight_x_parts,
                    weight_t_parts=cfg.loss.triplet.weight_t_parts,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'triplet_npairs_softmax':
                engine = torchreid.engine.ImageTripletNpairsSoftmaxFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_n=cfg.loss.triplet.weight_n,
                    weight_x_parts=cfg.loss.triplet.weight_x_parts,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    margin_sasc=cfg.loss.npairs.margin_sasc,
                    margin_sadc=cfg.loss.npairs.margin_sadc,
                    margin_dasc=cfg.loss.npairs.margin_dasc
                )
            elif cfg.loss.name == 'triplet_npairs_softmax_csattention':
                engine = torchreid.engine.ImageTripletNpairsSoftmaxCsattentionFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_n=cfg.loss.triplet.weight_n,
                    weight_x_parts=cfg.loss.triplet.weight_x_parts,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    margin_sasc=cfg.loss.npairs.margin_sasc,
                    margin_sadc=cfg.loss.npairs.margin_sadc,
                    margin_dasc=cfg.loss.npairs.margin_dasc
                )
            elif cfg.loss.name == 'triplet_npairs_softmax_separate':
                engine = torchreid.engine.ImageTripletNpairsSoftmaxSeparateFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_n=cfg.loss.triplet.weight_n,
                    weight_x_parts=cfg.loss.triplet.weight_x_parts,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    margin_sasc=cfg.loss.npairs.margin_sasc,
                    margin_sadc=cfg.loss.npairs.margin_sadc,
                    margin_dasc=cfg.loss.npairs.margin_dasc,
                    weight_separate=cfg.loss.separate.weight_separate
                )
            elif cfg.loss.name == 'triplet_npairs_dropbatch':
                engine = torchreid.engine.ImageTripletNpairsDropBatchFGnetEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_n=cfg.loss.triplet.weight_n,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    margin_sasc=cfg.loss.npairs.margin_sasc,
                    margin_sadc=cfg.loss.npairs.margin_sadc,
                    margin_dasc=cfg.loss.npairs.margin_dasc
                )
            else:
                exit("ERROR")
        else:
            if cfg.loss.name == 'softmax':
                engine = torchreid.engine.ImageSoftmaxEngine(
                    datamanager,
                    model,
                    optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'triplet_dropbatch':
                engine = torchreid.engine.ImageTripletDropBatchEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'group_triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageGroupTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_g=cfg.loss.triplet.weight_g,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_db_g=cfg.loss.dropbatch.weight_db_g,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    weight_b_db_g=cfg.loss.dropbatch.weight_b_db_g,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'dependency_triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageDependencyTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_dep=cfg.loss.triplet.weight_dep,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'focal_triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageFocalTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_f=cfg.loss.focal.weight_f,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_db_f=cfg.loss.focal.weight_db_f,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    weight_b_db_f=cfg.loss.focal.weight_b_db_f,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    gamma=cfg.loss.focal.gamma
                )
            elif cfg.loss.name == 'pose_triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImagePoseTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_mse=cfg.loss.pose.weight_mse,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_db_mse=cfg.loss.pose.weight_db_mse,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    weight_b_db_mse=cfg.loss.pose.weight_b_db_mse,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'cluster_triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageClusterTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_c=cfg.loss.cluster.weight_c,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_db_c=cfg.loss.cluster.weight_db_c,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    weight_b_db_c=cfg.loss.cluster.weight_b_db_c,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'cluster_dependency_triplet_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageClusterDependencyTripletDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_dep=cfg.loss.triplet.weight_dep,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_c=cfg.loss.cluster.weight_c,
                    weight_db_t=cfg.loss.dropbatch.weight_db_t,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_db_c=cfg.loss.cluster.weight_db_c,
                    weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    weight_b_db_c=cfg.loss.cluster.weight_b_db_c,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'reference_dropbatch_dropbotfeatures':
                engine = torchreid.engine.ImageReferenceDropBatchDropBotFeaturesEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_r=cfg.loss.triplet.weight_r,
                    weight_x=cfg.loss.triplet.weight_x,
                    weight_db_r=cfg.loss.dropbatch.weight_db_r,
                    weight_db_x=cfg.loss.dropbatch.weight_db_x,
                    weight_b_db_r=cfg.loss.dropbatch.weight_b_db_r,
                    weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                    top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            elif cfg.loss.name == 'triplet':
                engine = torchreid.engine.ImageTripletEngine(
                    datamanager,
                    model,
                    optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )
            else:
                exit("ERROR")
    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('-s', '--sources', type=str, nargs='+', help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+', help='target datasets (delimited by space)')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation')
    parser.add_argument('--root', type=str, default='', help='path to data root')
    parser.add_argument('--gpu-devices', type=str, default='',)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    if cfg.use_gpu and args.gpu_devices:
        # if gpu_devices is not specified, all available gpus will be used
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    datamanager = build_datamanager(cfg)
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        batch_num_classes=cfg.train.batch_size // cfg.sampler.num_instances
    )
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        args.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()