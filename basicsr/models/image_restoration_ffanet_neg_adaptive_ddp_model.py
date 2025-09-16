import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from .sr_neg_multi_model import SRNegModel_Multi
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ImageRestorationModel_FFANet_Neg_Multi_Adaptive_DDP(SRNegModel_Multi):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel_FFANet_Neg_Multi_Adaptive_DDP, self).__init__(opt)
        self.mean = torch.tensor(
            [0.64, 0.6, 0.58]).reshape(-1, 3, 1, 1).float().to(self.device)
        self.std = torch.tensor(
            [0.14, 0.15, 0.152]).reshape(-1, 3, 1, 1).float().to(self.device)

        train_opt = opt.get('train', None)
        self.neg_t = train_opt.get('neg_t', 0.022)

        if train_opt is not None:
            if train_opt.get('perceptual_pos_opt'):
                self.cri_perceptual_pos = build_loss(
                    train_opt['perceptual_pos_opt']).to(self.device)
            else:
                self.cri_perceptual_pos = None

            if train_opt.get('perceptual_contrastive_opt'):
                self.cri_perceptual_contrastive = build_loss(
                    train_opt['perceptual_contrastive_opt']).to(self.device)
            else:
                self.cri_perceptual_contrastive = None

        else:
            pass
        self.update_neg_count = 0

    def feed_data(self, data, is_val=False):
        lq = (data['lq'].to(self.device) - self.mean)/self.std
        self.lq = lq
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * \
            self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil(
            (w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil(
            (h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale:(i + crop_size_h) //
                             scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def optimize_parameters(self, current_iter=None, tb_logger=None):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        with torch.no_grad():
            neg_out = [net(self.lq) for net in self.net_g_neg]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_neg = 0.0

            for neg_out_ in neg_out:
                l_percep, l_style = self.cri_perceptual(self.output, neg_out_)
                l_neg += l_percep

            if l_percep is not None:
                l_neg = l_neg/self.neg_num
                l_total -= l_neg
                loss_dict['l_neg'] = l_neg

            with torch.no_grad():
                std_v, mean_v = torch.std_mean(
                    2*(self.output-self.gt), dim=[-3, -2, -1])
                std_v = std_v.mean()
                mean_v = mean_v.mean()
                if self.opt['rank'] == 0:
                    if tb_logger:
                        tb_logger.add_scalar(
                            f'train/grad/mean', mean_v, current_iter)
                        tb_logger.add_scalar(
                            f'train/grad/std', std_v, current_iter)
                if std_v > self.neg_t:
                    update_neg_id = self.update_neg_count % self.neg_num
                    self.model_neg(self.net_g_neg[update_neg_id], 0.001)
                    self.update_neg_count += 1
                    logger = get_root_logger()
                    logger.warning(
                        f'Update Negative Model {update_neg_id}, total {self.update_neg_count}')

        if self.cri_perceptual_pos:
            l_percep, l_style = self.cri_perceptual_pos(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_pos'] = l_percep

        if self.cri_perceptual_contrastive:
            l_neg = 0.0

            for neg_out_ in neg_out:
                l_percep = self.cri_perceptual_contrastive(
                    self.output, neg_out_, self.gt)
                l_neg += l_percep

            if l_percep is not None:
                l_neg = l_neg/self.neg_num
                l_total += l_neg
                loss_dict['l_contrastive'] = l_neg

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', False)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(), use_grad_clip)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # for idx, neg_iter in enumerate(self.neg_iter[::-1]):
        #     if current_iter % neg_iter == 0:
        #         self.model_neg(self.net_g_neg[idx], decay=self.neg_decay[idx])

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * \
            self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j +
                  crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        metric_data = dict()

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(
                        self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])

                for name, opt_ in opt_metric.items():
                    self.metric_results[name] += calculate_metric(
                        metric_data, opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(
                    self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(
                cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.warning(
            'nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
            tb_logger.add_scalar(f'train/loss/{metric}', value, current_iter)

        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
