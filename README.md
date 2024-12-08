# Structure-Aware Human Body Reshaping with Adaptive Affinity-Graph Network (WACV 2025)
**Qiwen Deng<sup>1</sup>\*, Yangcen Liu<sup>2</sup>\*** (*Equal contribution, order decided by coin toss*)

- **Affiliations**:  
  1. University of Electronic Science and Technology of China  
  2. Georgia Institute of Technology  


**[Paper Link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Xq-r3dIAAAAJ&citation_for_view=Xq-r3dIAAAAJ:d1gkVwhDpl0C)**  

## Overview  
---

![Pipeline](gif/pipe.png)

This repository provides the implementation for **Adaptive Affinity-Graph Network (AAGN)**, a novel approach for automatic human body reshaping. The automatic human body reshaping task focuses on transforming a given portrait into an aesthetically enhanced body shape. AAGN enhances global consistency and aesthetic quality. Adaptive Affinity-Graph (AAG) Block: Captures global affinities between body parts for consistent reshaping. Body Shape Discriminator (BSD): Focuses on high-frequency details for improved aesthetics.

Here are visual results showcasing the transformations achieved by AAGN:  

<table>
  <tr>
    <td><img src="gif/1.gif" alt="GIF 1" width="200"></td>
    <td><img src="gif/2.gif" alt="GIF 2" width="200" ></td>
    <td><img src="gif/3.gif" alt="GIF 3" width="200"></td>
    <td><img src="gif/4.gif" alt="GIF 4" width="200"></td>
  </tr>
  <tr>
    <td><img src="gif/5.gif" alt="GIF 5" width="200"></td>
    <td><img src="gif/6.gif" alt="GIF 6" width="200"></td>
    <td><img src="gif/7.gif" alt="GIF 7" width="200"></td>
    <td><img src="gif/8.gif" alt="GIF 8" width="200"></td>
  </tr>
</table>

## BR5K Dataset
--- 

We utilize the **BR-5K dataset**, the largest dataset for human body reshaping tasks. The dataset preparation process follows the guidelines provided by the [FBBR repository](https://github.com/JianqiangRen/FlowBasedBodyReshaping?tab=readme-ov-file).   


## Get Started
---
### Install Requirements

&#8226; python >= 3.7
&#8226; torch >= 1.2.0

### Pretrained Models

Our checkpoint:

### Run the Demo
<pre>
<code>
python demo.py --dir
</code>
</pre>

### Training
<pre>
<code>
python train.py
</code>
</pre>

### Evaluation
To evaluate:
<pre>
<code>
python evaluate.py --
</code>
</pre>

| Method       | SSIM ↑   | PSNR ↑   | LPIPS ↓   |
|--------------|----------|----------|-----------|
| GFLA         | 0.6649   | 21.4796  | 0.6136    |
| pix2pixHD    | 0.7271   | 21.8381  | 0.2800    |
| FAL          | 0.8261   | 24.1841  | 0.0837    |
| ATW          | 0.8316   | 24.6332  | 0.0805    |
| FBBR         | 0.8354   | 24.7924  | 0.0777    |
| **Ours**     | **0.8427** | **26.4100** | **0.0643** |



## Citation  
---

If you find our work helpful in your research, please consider citing us:  

```bibtex
@misc{deng2024structureawarehumanbodyreshaping,
      title={Structure-Aware Human Body Reshaping with Adaptive Affinity-Graph Network}, 
      author={Qiwen Deng and Yangcen Liu and Wen Li and Guoqing Wang},
      year={2024},
      eprint={2404.13983},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.13983} 
}
```

## Acknowledgement  
---  
We express our gratitude to [FBBR](https://github.com/JianqiangRen/FlowBasedBodyReshaping?tab=readme-ov-file) as we benefited greatly from their paper and code.  
