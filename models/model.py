import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import numpy as np
import os
import cv2
import scipy.io as sio
from skimage import img_as_ubyte

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old*self.beta+(1-self.beta)*new

class NoiseDA(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        super(NoiseDA, self).__init__(**kwargs)

        self.loss_fn = losses
        self.netG = networks[0]
        self.warmup = False

        if ema_scheduler is not None:                
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay']) 
        else:
            self.ema_scheduler = None
        
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_trainings()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

    def set_input_train(self, data):
        self.syn_img = self.set_device(data.get('syn_img'))
        self.gt_img = self.set_device(data.get('gt_img'))
        self.real_img = self.set_device(data.get('real_img'))
        self.ref_img = self.set_device(data.get('ref_img'))
        self.path = data['path']
        self.batch_size = len(data['path'])
        
    def set_input_test(self, data):
        self.input_img = self.set_device(data.get('input_img'))
        self.gt_img = self.set_device(data.get('gt_img'))
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        clamp_min, clamp_max = (0., 1.) if self.norm == 'norm0' else (-1., 1.)
        if phase == 'train':
            dict = {
                'gt_img': torch.clamp(self.gt_img.detach()[:], clamp_min, clamp_max).float().cpu(),
                'syn_img': torch.clamp(self.syn_img.detach()[:], clamp_min, clamp_max).float().cpu(),
                'real_img': torch.clamp(self.real_img.detach()[:], clamp_min, clamp_max).float().cpu(),
                'ref_img': torch.clamp(self.ref_img.detach()[:], clamp_min, clamp_max).float().cpu()
            }
        if phase == 'val' or phase == 'test':
            dict = {
                'gt_img': torch.clamp(self.gt_img.detach()[:], clamp_min, clamp_max).float().cpu(),
                'input_img': torch.clamp(self.input_img.detach()[:], clamp_min, clamp_max).float().cpu(),
            }

        if phase != 'train':
            dict.update({
                'output': torch.clamp(self.output.detach()[:], clamp_min, clamp_max).float().cpu()
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('{}'.format(self.path[idx]))
            ret_result.append(self.output[idx].detach().float().cpu())
            
        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self, epoch):
        self.netG.train()
        self.train_metrics.reset()
        p = min(float(epoch/self.ctrl_epoch), 1.)
        alpha = (2./(1.+np.exp(-5*p))-1)*self.diff_weight
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input_train(train_data)
            self.optG.zero_grad()
            
            if(self.diff_flag==0):
                '''train the vanilla restoration net'''
                loss_syn = self.netG(self.gt_img, self.syn_img, diff_flag=self.diff_flag)
                loss_syn.backward()
                self.optG.step()

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='train')
                self.train_metrics.update(self.loss_fn[1].__name__, loss_syn.item())
            else:
                '''train the restoration net with diffusion'''
                loss_noise, loss_syn = self.netG(self.gt_img, self.syn_img, self.real_img, self.ref_img, diff_flag=self.diff_flag)
                loss_noise_alpha = alpha * loss_noise
                loss = loss_noise_alpha + loss_syn
                loss.backward()
                self.optG.step()

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='train')
                self.train_metrics.update(self.loss_fn[0].__name__, loss_noise.item())
                self.train_metrics.update(self.loss_fn[1].__name__, loss_syn.item())
                self.train_metrics.update(self.loss_fn[2].__name__, loss_noise_alpha.item())
                self.train_metrics.update(self.loss_fn[3].__name__, loss.item())
                            
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input_test(val_data)
                
                if self.opt['distributed']:
                    self.output = self.test_split_fn(self.netG.module, self.input_img)
                else:
                    self.output = self.test_split_fn(self.netG, self.input_img)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')
                
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_img, self.output, self.norm)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        
        if(self.real_denoise):
            '''test on the denosing task using real-world SIDD test set'''
            with torch.no_grad():
                for phase_data in tqdm.tqdm(self.phase_loader):
                    self.set_input_test(phase_data)
                    
                    for i in range(40):
                        for k in range(32):
                            noisy_patch = self.input_img[:,i,k,:,:,:].permute(0,3,1,2)
                            if self.opt['distributed']:
                                restored_patch = self.test_split_fn(self.netG.module, noisy_patch)
                            else:
                                restored_patch = self.test_split_fn(self.netG, noisy_patch)
                            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0,2,3,1).squeeze(0)
                            self.gt_img[:,i,k,:,:,:] = restored_patch
                            save_file = os.path.join(self.save_path, '%04d_%02d.png'%(i+1,k+1))
                            cv2.imwrite(save_file, img_as_ubyte(restored_patch))
                    
                    sio.savemat(os.path.join(self.save_path, 'Idenoised.mat'), {"Idenoised": self.gt_img.detach().cpu().numpy()[0],})

        else:
            '''test on other image restortation datasets such as derain, deblur'''
            with torch.no_grad():
                for phase_data in tqdm.tqdm(self.phase_loader):
                    self.set_input_test(phase_data)
                    
                    if self.opt['distributed']:
                        self.output = self.test_split_fn(self.netG.module, self.input_img)
                    else:
                        self.output = self.test_split_fn(self.netG, self.input_img)

                    self.iter += self.batch_size
                    self.writer.set_iter(self.epoch, self.iter, phase='test')
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_img, self.output, self.norm)
                        self.test_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results())

            test_log = self.test_metrics.result()
            ''' save logged informations into log dict '''
            test_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard '''
            for key, value in test_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
    def load_networks(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
            
    def resume_trainings(self):
        """ resume training from the latest checkpoint. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.resume_training(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.resume_training(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
    
    def test_split_fn(self, model, L, dep_U=4, refield=128, min_size=2500, sf=1, modulo=1):
        """
        Args:
            model: trained model
            L: input Low-quality image
            refield: effective receptive filed of the network, 32 is enough
            min_size: min_sizeXmin_size image, e.g., 256X256 image
            sf: scale factor for super-resolution, otherwise 1
            modulo: 1 if split

        Returns:
            E: estimated result
        """
        h, w = L.size()[-2:]
        if h*w <= min_size**2:
            lenU = 2 ** (dep_U-1)
            padH = 0 if ((h % lenU) == 0) else (lenU - (h % lenU))
            padW = 0 if ((w % lenU) == 0) else (lenU - (w % lenU))
            L = torch.nn.ReplicationPad2d((0, padW, 0, padH))(L)
            E = model.restoration_test_head(L)
            E = E[:, :, :h, :w]
        else:
            top = slice(0, (h//2//refield+1)*refield)
            bottom = slice(h - (h//2//refield+1)*refield, h)
            left = slice(0, (w//2//refield+1)*refield)
            right = slice(w - (w//2//refield+1)*refield, w)
            Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

            if h * w <= 4*(min_size**2):
                Es = [model.restoration_test_head(Ls[i]) for i in range(4)]
            else:
                Es = [self.test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

            b, c = Es[0].size()[:2]
            E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

            E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
            E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
            E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
            E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
        return E