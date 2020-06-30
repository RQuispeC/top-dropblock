from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results', 'visualize_ranked_activation_results', 'visualize_ranked_threshold_activation_results', 'visualize_ranked_mask_activation_results']

import numpy as np
import os
import os.path as osp
import shutil
import cv2
from matplotlib import pyplot as plt

from .tools import mkdir_if_missing


GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        
        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones((height, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(save_dir, osp.basename(osp.splitext(qimg_path_name)[0]))
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            
            if not invalid:
                matched = gpid==qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx+1)*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:, start: end, :] = gimg
                else:
                    _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery', matched=matched)
                
                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))


def visualize_ranked_activation_results(distmat, query_act, gallery_act, dataset, data_type, width=128, height=256, save_dir='', topk=10):
    """Visualizes ranked results with activation maps.

    Supports only image-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        query_act (torch tensor): activations for query (num_query)
        gallery_act (torch tensor): activations for gallery (num_gallery)
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    if data_type != 'image':
        raise KeyError("Unsupported data type: {}".format(data_type))
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones((2*height+10, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)

            qact = query_act[q_idx].numpy()
            qact = np.uint8(np.floor(qact))
            qact = cv2.applyColorMap(qact, cv2.COLORMAP_JET)
            overlapped = qimg * 0.5 + qact * 0.5
            overlapped[overlapped>255] = 255
            overlapped = overlapped.astype(np.uint8)
            grid_img[:height, :width, :] = qimg
            grid_img[height+10:, :width, :] = overlapped
        else:
            pass

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid==qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx+1)*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING

                    gact = gallery_act[g_idx].numpy()
                    gact = np.uint8(np.floor(gact))
                    gact = cv2.applyColorMap(gact, cv2.COLORMAP_JET)
                    overlapped = gimg * 0.5 + gact * 0.5
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)
                    grid_img[:height, start: end, :] = gimg
                    grid_img[height+10:, start: end, :] = overlapped
                else:
                    pass

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))

def visualize_ranked_threshold_activation_results(distmat, query_act, gallery_act, dataset, data_type, width=128, height=256, save_dir='', topk=10, threshold=0.7):
    """Visualizes ranked results with activation maps.

    Supports only image-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        query_act (torch tensor): activations for query (num_query)
        gallery_act (torch tensor): activations for gallery (num_gallery)
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    if data_type != 'image':
        raise KeyError("Unsupported data type: {}".format(data_type))
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        qact = query_act[q_idx].numpy()

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones((3*height+20, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)

            qact = query_act[q_idx].numpy()
            qact = np.uint8(np.floor(qact))
            mask = np.zeros_like(qact)
            mask[(qact/255)>=threshold] = 255
            mask = mask.astype(np.uint8)
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)
            qact = cv2.applyColorMap(qact, cv2.COLORMAP_JET)
            overlapped = qimg * 0.5 + qact * 0.5
            overlapped[overlapped>255] = 255
            overlapped = overlapped.astype(np.uint8)
            grid_img[:height, :width, :] = qimg
            grid_img[height+10: 2*height+10, :width, :] = overlapped
            grid_img[2*height+20:, :width, :] = mask

        else:
            pass

        rank_idx = 1
        matched_first = False
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid==qpid
                if rank_idx==1:#save if rank-1 is correct
                    matched_first = matched
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx+1)*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING

                    gact = gallery_act[g_idx].numpy()
                    gact = np.uint8(np.floor(gact))
                    mask = np.zeros_like(gact)
                    mask[(gact/255)>=threshold] = 255
                    mask = mask.astype(np.uint8)
                    mask = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)
                    gact = cv2.applyColorMap(gact, cv2.COLORMAP_JET)
                    overlapped = gimg * 0.5 + gact * 0.5
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)
                    grid_img[:height, start: end, :] = gimg
                    grid_img[height+10: 2*height+10, start: end, :] = overlapped
                    grid_img[2*height+20:, start: end, :] = mask
                else:
                    pass

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            if not matched_first: imname = "error_" + imname
            cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))


def visualize_ranked_mask_activation_results(distmat, query_act, gallery_act, query_mask, gallery_mask, dataset, data_type, width=128, height=256, save_dir='', topk=10, threshold=0.7):
    """Visualizes ranked results with activation maps.

    Supports only image-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        query_act (torch tensor): activations for query (num_query)
        gallery_act (torch tensor): activations for gallery (num_gallery)
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    if data_type != 'image':
        raise KeyError("Unsupported data type: {}".format(data_type))
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing dropmask for top-{} ranks ...'.format(topk))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        qact = query_act[q_idx].numpy()

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones((3*height+20, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)

            qact = query_act[q_idx].numpy()
            qact = np.uint8(np.floor(qact))
            mask = query_mask[q_idx].numpy()
            mask = mask.astype(np.uint8)
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)
            qact = cv2.applyColorMap(qact, cv2.COLORMAP_JET)
            overlapped = qimg * 0.5 + qact * 0.5
            overlapped[overlapped>255] = 255
            overlapped = overlapped.astype(np.uint8)
            overlapped_mask = qimg * mask
            overlapped_mask = overlapped_mask.astype(np.uint8)
            grid_img[:height, :width, :] = qimg
            grid_img[height+10: 2*height+10, :width, :] = overlapped
            grid_img[2*height+20:, :width, :] = overlapped_mask

        else:
            pass

        rank_idx = 1
        matched_first = False
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid==qpid
                if rank_idx==1:#save if rank-1 is correct
                    matched_first = matched
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx+1)*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING

                    gact = gallery_act[g_idx].numpy()
                    gact = np.uint8(np.floor(gact))
                    mask = gallery_mask[g_idx].numpy()
                    mask = mask.astype(np.uint8)
                    mask = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)
                    gact = cv2.applyColorMap(gact, cv2.COLORMAP_JET)
                    overlapped = gimg * 0.5 + gact * 0.5
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)
                    overlapped_mask = gimg * mask
                    overlapped_mask = overlapped_mask.astype(np.uint8)
                    grid_img[:height, start: end, :] = gimg
                    grid_img[height+10: 2*height+10, start: end, :] = overlapped
                    grid_img[2*height+20:, start: end, :] = overlapped_mask
                else:
                    pass

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            if not matched_first: imname = "error_" + imname
            cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
