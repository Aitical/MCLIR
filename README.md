<div align="center">

# Learning from History: Task-agnostic Model Contrastive Learning for Image Restoration

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun), [Kui Jiang](https://github.com/kuijiang94), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Harbin Institute of Technology, Harbin 150001, China.

</div>


> Contrastive learning has emerged as a prevailing paradigm for high-level vision tasks, which, by introducing properly negative samples, has also been exploited for low-level vision tasks to achieve a compact optimization space to account for their ill-posed nature. However, existing methods rely on manually predefined, task-oriented negatives, which often exhibit pronounced task-specific biases. In this paper, we propose a innovative approach for the adaptive generation of negative samples directly from the target model itself, called *learning from history*. We introduce the Self-Prior guided Negative loss for image restoration (SPNIR) to enable this approach. Our approach is task-agnostic and generic, making it compatible with any existing image restoration method or task. We demonstrate the effectiveness of our approach by retraining existing models with SPNIR. The results show significant improvements in image restoration across various tasks and architectures. For example, models retrained with SPNIR outperform the original FFANet and DehazeFormer by 3.41 dB and 0.57 dB on the RESIDE indoor dataset for image dehazing. Similarly, they achieve notable improvements of 0.47 dB on SPA-Data over IDT for image deraining and 0.12 dB on Manga109 for a 4x scale super-resolution over lightweight SwinIR, respectively. 

## Model Contrastive Paradigm for Image Restoration
<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2023/08/15/frameworkv2_new.jpeg" alt="frameworkv2_new.jpeg" border="0" />
</div>

<br />

### Compared to Previous Methods


| Methods | Task & Dataset | PSNR | SSIM |
| :---: | :---: | :---: | :---: |
| FFANet (Qin et al. 2020b) |  | 36.39 | 0.9886 |
| +CR (Wu et al. 2021) | Image Dehazing | 36.74 | 0.9906 |
| +CCR (Zheng et al. 2023) | (SOTS-indoor) | 39.24 | 0.9937 |
| **+SPN (Ours)** |  | <font color='red'><b>39.80</b></font>|  <font color='red'><b>0.9947</b> </font>|
| EDSR (Lim et al. 2017) |  | 26.04 | 0.7849 |
| +PCL (Wu, Jiang, and Liu 2023) | SISR | 26.07 | 0.7863 |
| **+SPN (Ours)** | (Urban100) | <font color='red'><b>26.12</b> </font>| <font color='red'><b>0.7878</b></font> |

<br />

### Easy to Follow


There is a simple implementtation of our _Model Contrastive Paradigm_ and **Self-Prior Guided Negative Loss**.

```diff
-def train_iter(model, lq_input, hq_output, current_iter)
+def train_iter(model, negative_model, lq_input, hq_output, current_iter, lambda, update_step):
    optimizer.zero_grad()
    output = target_model(lq_input)
    L_rec = l1_loss(output, hq_gt)

+   ## Add Negative Sample
+   neg_sample = negative_model(lq_input)
+   ## Add Negative Loss
+   L_neg = perceptual_vgg_loss(output, neg_sample)
+   Loss = L_rec + lammbda * L_neg
    Loss.backward()
    optimizer.step()
+   if current_iter % update_step == 0:
+    update_model_ema(negative_model, target_model)

```

With just a few modifications to your own training scripts, you can easily integrate our approach. Enjoy it!


## Results
<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2023/08/15/improvementv2.jpeg" alt="improvementv2.jpeg" border="0" />
</div>
<br />


### Image Super-Resolution
<small>
<table>
    <tr>
        <td>Method</td>
        <td>Architecture</td>
        <td>Scale</td>
        <td>Avg.</td>
        <td>Set14</td>
        <td>B100</td>
        <td>Urban100</td>
        <td>Manga109</td>
    </tr>
    <tr>
        <td> </td>
        <td> </td>
        <td> </td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
    </tr>
    <tr>
        <td>EDSR-light</td>
        <td> </td>
        <td>x2</td>
        <td>32.06/0.9303</td>
        <td>33.57/0.9175</td>
        <td>32.16/0.8994</td>
        <td>31.98/0.9272</td>
        <td>30.54/0.9769</td>
    </tr>
    <tr>
        <td><b>+SPN (Ours)</b></td>
        <td>CNN</td>
        <td> </td>
        <td><b>32.19/0.9313</b></td>
        <td><b>33.67/0.9182</b></td>
        <td><b>32.21/0.9001</b></td>
        <td><b>32.23/0.9297</b></td>
        <td><b>30.64/0.9772</b></td>
    </tr>
    <tr>
        <td>EDSR-light</td>
        <td> </td>
        <td>x4</td>
        <td>28.14/0.8021</td>
        <td>28.58/0.7813</td>
        <td>27.57/0.7357</td>
        <td>26.04/0.7849</td>
        <td>30.35/0.9067</td>
    </tr>
    <tr>
        <td><b>+SPN (Ours)</b></td>
        <td> </td>
        <td> </td>
        <td><b>28.21/0.8040</b></td>
        <td><b>28.63/0.7829</b></td>
        <td><b>27.59/0.7369</b></td>
        <td><b>26.12/0.7878</b></td>
        <td><b>30.51/0.9085</b></td>
    </tr>
    <tr>
        <td>SwinIR-light</td>
        <td> </td>
        <td>x4</td>
        <td>28.46/0.8099</td>
        <td>28.77/0.7858</td>
        <td>27.69/0.7406</td>
        <td>26.47/0.7980</td>
        <td>30.92/0.9151</td>
    </tr>
    <tr>
        <td><b>+SPN (Ours)</b></td>
        <td>Transformer</td>
        <td> </td>
        <td><b>28.55/0.8114</b></td>
        <td><b>28.85/0.7874</b></td>
        <td><b>27.72/0.7414</b></td>
        <td><b>26.57/0.8010</b></td>
        <td><b>31.04/0.9158</b></td>
    </tr>
    <tr>
        <td>SwinIR</td>
        <td> </td>
        <td>x4</td>
        <td>28.88/0.8190</td>
        <td>28.94/0.7914</td>
        <td>27.83/0.7459</td>
        <td>27.07/0.8164</td>
        <td>31.67/0.9226</td>
    </tr>
    <tr>
        <td><b>+SPN (Ours)</b></td>
        <td> </td>
        <td> </td>
        <td><b>28.93/0.8198</b></td>
        <td><b>29.01/0.7923</b></td>
        <td><b>27.85/0.7465</b></td>
        <td><b>27.14/0.8176</b></td>
        <td><b>31.75/0.9229</b></td>
    </tr>
</table>
</small>
<br />


<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2023/08/15/SR_vision.jpeg" alt="SR_vision.jpeg" border="0" />
</div>


<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2023/08/15/SR_vision_More.jpeg" alt="SR_vision_More.jpeg" border="0" />
</div>
<br />

### Image Hehazing

<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2023/08/15/FFANet_Vision.jpeg" alt="FFANet_Vision.jpeg" border="0" />
</div>

<br />

<table>
    <tr>
        <td>Methods</td>
        <td>SOTS-indoor</td>
        <td> </td>
        <td>SOTS-mix</td>
        <td> </td>
    </tr>
    <tr>
        <td> </td>
        <td>PSNR</td>
        <td>SSIM</td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>(ICCV'19) GridDehazeNet</td>
        <td>32.16</td>
        <td>0.984</td>
        <td>25.86</td>
        <td>0.944</td>
    </tr>
    <tr>
        <td>(CVPR'20) MSBDN</td>
        <td>33.67</td>
        <td>0.985</td>
        <td>28.56</td>
        <td>0.966</td>
    </tr>
    <tr>
        <td>(ECCV'20) PFDN</td>
        <td>32.68</td>
        <td>0.976</td>
        <td>28.15</td>
        <td>0.962</td>
    </tr>
    <tr>
        <td>(AAAI'20) FFANet</td>
        <td>36.39</td>
        <td>0.989</td>
        <td>29.96</td>
        <td>0.973</td>
    </tr>
    <tr>
        <td><b>(Ours) FFANet+SPN</b></td>
        <td><b>39.80</b></td>
        <td><b>0.995</b></td>
        <td><b>30.65</b></td>
        <td><b>0.976</b></td>
    </tr>
    <tr>
        <td>(TIP'23) DehazeFormer-T</td>
        <td>35.15</td>
        <td>0.989</td>
        <td>30.36</td>
        <td>0.973</td>
    </tr>
    <tr>
        <td><b>(Ours) DehazeFormer-T+SPN</b> </td>
        <td><b>35.51</b></td>
        <td><b>0.990</b></td>
        <td><b>30.44</b></td>
        <td><b>0.974</b></td>
    </tr>
    <tr>
        <td>(TIP'23)DehazeFormer-S</td>
        <td>36.82</td>
        <td>0.992</td>
        <td>30.62</td>
        <td>0.976</td>
    </tr>
    <tr>
        <td><b>(Ours) DehazeFormer-S+SPN </b></td>
        <td><b>37.24</b></td>
        <td><b>0.993</b></td>
        <td><b>30.77</b></td>
        <td><b>0.978</b></td>
    </tr>
    <tr>
        <td>(TIP'23) DehazeFormer-B</td>
        <td>37.84</td>
        <td>0.994</td>
        <td>31.45</td>
        <td>0.980</td>
    </tr>
    <tr>
        <td><b>(Ours) DehazeFormer-B+SPN </b></td>
        <td><b>38.41</b></td>
        <td><b>0.994</b></td>
        <td><b>31.57</b> </td>
        <td><b>0.981</b></td>
    </tr>
</table>
<br />


### Image Deraining
<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2023/08/17/rain_vision.jpeg" alt="rain_vision.jpeg" border="0" />
<small>A divergence map delineates the differences, highlighting the improvement achieved by ours, particularly in degraded regions. </small>
</div>
<br />
<small>
<table>
    <tr>
        <td>Method</td>
        <td>Avg.</td>
        <td>Rain100L</td>
        <td>Rain100H</td>
        <td>DID</td>
        <td>DDN</td>
        <td>SPA</td>
    </tr>
    <tr>
        <td>undefined</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
        <td>PSNR/SSIM</td>
    </tr>
    <tr>
        <td>(CVPR21) MPRNet</td>
        <td>36.17/0.9543</td>
        <td>39.47/0.9825</td>
        <td>30.67/0.9110</td>
        <td>33.99/0.9590</td>
        <td>33.10/0.9347</td>
        <td>43.64/0.9844</td>
    </tr>
    <tr>
        <td>(AAAI'21) DualGCN</td>
        <td>36.69/0.9604</td>
        <td>40.73/0.9886</td>
        <td>31.15/0.9125</td>
        <td>34.37/0.9620</td>
        <td>33.01/0.9489</td>
        <td>44.18/0.9902</td>
    </tr>
    <tr>
        <td>(ICCV'21) SPDNet</td>
        <td>36.54/0.9594</td>
        <td>40.50/0.9875</td>
        <td>31.28/0.9207</td>
        <td>34.57/0.9560</td>
        <td>33.15/0.9457</td>
        <td>43.20/0.9871</td>
    </tr>
    <tr>
        <td>(ICCV'21) SwinIR</td>
        <td>36.91/0.9507</td>
        <td>40.61/0.9871</td>
        <td>31.76/0.9151</td>
        <td>34.07/0.9313</td>
        <td>33.16/0.9312</td>
        <td>44.97/0.9890</td>
    </tr>
    <tr>
        <td>(CVPR'22) Uformer-S</td>
        <td>36.95/0.9505</td>
        <td>40.20/0.9860</td>
        <td>30.80/0.9105</td>
        <td>34.46/0.9333</td>
        <td>33.14/0.9312</td>
        <td>46.13/0.9913</td>
    </tr>
    <tr>
        <td>(CVPR'22) Restormer</td>
        <td>37.49/0.9530</td>
        <td>40.58/0.9872</td>
        <td>31.39/0.9164</td>
        <td>35.20/0.9363</td>
        <td>34.04/0.9340</td>
        <td>46.25/0.9911</td>
    </tr>
    <tr>
        <td>(TPAMI'23) IDT</td>
        <td>37.77/0.9593</td>
        <td>40.74/0.9884</td>
        <td>32.10/0.9343</td>
        <td>34.85/0.9401</td>
        <td>33.80/0.9407</td>
        <td>47.34/0.9929</td>
    </tr>
    <tr>
        <td><b>(Ours) IDT+SPN</b></td>
        <td><b>38.03/0.9610</b></td>
        <td><b>41.12/0.9893</b></td>
        <td><b>32.17/0.9352</b></td>
        <td><b>34.94/0.9424</b></td>
        <td><b>33.90/0.9442</b></td>
        <td><b>48.04/0.9938</b></td>
    </tr>
</table>
</small>

<br />

### Image Deblurring


<div style="text-align: center">

<table>
    <tr>
        <td>Method</td>
        <td>MIMO-UNet</td>
        <td>HINet</td>
        <td>MAXIM</td>
        <td>Restormer</td>
        <td>UFormer</td>
        <td>NAFNet</td>
        <td><b>NAFNet+SPN (Ours)</b></td>
    </tr>
    <tr>
        <td>PSNR</td>
        <td>32.68</td>
        <td>32.71</td>
        <td>32.86</td>
        <td>32.92</td>
        <td>32.97</td>
        <td>32.87</td>
        <td><b>32.93</b></td>
    </tr>
    <tr>
        <td>SSIM</td>
        <td>0.959</td>
        <td>0.959</td>
        <td>0.961</td>
        <td>0.961</td>
        <td>0.967</td>
        <td>0.9606</td>
        <td><b>0.9619</b></td>
    </tr>
</table>
</div>

<br />


### Retrained Models

|EDSR baseline|SwinIR-light| SwinIR-Large| FFANet|
| :---: | :---: | :---: | :---: |
|[Download](https://drive.google.com/file/d/1U4HkdIgqBWcNnRBrdvfzhfVFvbz8vWLq/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1mXXbsUBxQMiBO0A3BS7i_MD8TswU6TmA/view?usp=drive_link)|[Download](https://drive.google.com/file/d/12FooL4KwL0TtqBNFuksV2Hjk82RTUIno/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1vm9wTFJXOZWVFSIpBZSp0jl1cbK6a8_7/view?usp=drive_link)|

**Quick Evaluation Guide**

For quickly evaluating, download the retrained models enhanced by our Model Contrastive Learning. The test scripts for each model are available in their respective repositories:
[BasicSR](https://github.com/XPixelGroup/BasicSR), [FFANet](https://github.com/zhilin007/FFA-Net), [DehazeFormer](https://github.com/IDKiro/DehazeFormer/tree/main), [IDT](https://github.com/jiexiaou/IDT), and [NAFNet](https://github.com/megvii-research/NAFNet).
Our gratitude goes out to the authors for their nice sharing of these projects.



## Reference
- 	Xu Qin, Zhilin Wang, Yuanchao Bai, Xiaodong Xie, Huizhu Jia:
FFA-Net: Feature Fusion Attention Network for Single Image Dehazing. AAAI 2020: 11908-11915
- 	Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee:
Enhanced Deep Residual Networks for Single Image Super-Resolution. CVPR Workshops 2017: 1132-1140
- G. Wu, J. Jiang and X. Liu, "A Practical Contrastive Learning Framework for Single-Image Super-Resolution," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2023.3290038
        
        
        
        
        
        
- Haiyan Wu, Yanyun Qu, Shaohui Lin, Jian Zhou, Ruizhi Qiao, Zhizhong Zhang, Yuan Xie, Lizhuang Ma:
Contrastive Learning for Compact Single Image Dehazing. CVPR 2021: 10551-10560
- Yu Zheng, Jiahui Zhan, Shengfeng He, Junyu Dong, Yong Du:
Curricular Contrastive Regularization for Physics-aware Single Image Dehazing. CVPR 2023
