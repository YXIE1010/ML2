# ML2: Machine Learning based Melting Layer detection
### Leverage machine learning for automatic detection of melting layers from profiling radar observations     

> This project is currently being written into a journal article to be submitted to [*Journal of Geophysical Research: Machine Learning and Computation*](https://agupubs.onlinelibrary.wiley.com/journal/29935210)

## Key points
- A binary semantic segmentation U-Net model has been developed to detect boundary heights and thickness of melting layers using Ka-band ground profiling radar observations at the North Slope of Alaska site.

- The U-Net model is trained by manually labeled melting layers collected via an interactive data extraction tool, named ClickCollect, which is a by-product of this ML2 project but can be applied to a wide range of applications. ClickCollect is made publicly available [ClickCollect GitHub repository](https://github.com/YXIE1010/ClickCollect).

- Compared to a traditional detection method, the U-Net model detects melting layers more effectively and accurately. Moreover, the U-Net model offers better general applicability and performs well under various weather conditions including **heavy precipitation with radar velocity folding**, **multi-layer melting process**, and **shallow melting layers**.

- The U-Net model shows promise in providing interpretable detections with uncertainty estimation based on ensemble predictions.

## Hands-on example
*File in preparation, Stay tuned*

## Authors
Yan Xie (yanxieyx@umich.edu), Fraser King, Claire Pettersen, Mark Flanner   
University of Michigan, Ann Arbor, MI 48105




