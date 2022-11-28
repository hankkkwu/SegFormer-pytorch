Building the blocks of [Segformer](https://arxiv.org/abs/2105.15203) architecture.
1. **Overlap Patch Embedding**. A method to convert images to sequence of overlapping patches
2. **Efficient Self-Attention** - 1st Core component of all Transformer based models.
3. **Mix-FeedForward module** - 2nd core component of Transformer models. Along with Self-Attention, forms single Transformer block
4. **Transformer block** - Self-attention + Mix FFN + Layer Norm forms a basic Tranformer block5.
5. **Decoder head** - contains MLP layers.

Here is the [result](https://www.youtube.com/watch?v=O-6A58Y2PvI "semantic segmentation") trained on BDD100k drivable area:
[![highway-seg](https://github.com/hankkkwu/SegFormer-pytorch/blob/main/results/highway_seg.gif)](https://www.youtube.com/watch?v=O-6A58Y2PvI "semantic segmentation")

Here is the [attention maps](https://www.youtube.com/watch?v=lfDrSyx-jQY "attention maps") from the video above:
[![highway-attn](https://github.com/hankkkwu/SegFormer-pytorch/blob/main/results/highway_attn.gif)](https://www.youtube.com/watch?v=lfDrSyx-jQY "attention maps")